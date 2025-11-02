"""
Data Processing Module for Wyoming Mobility Analysis
"""

import pandas as pd
import json
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

from advan_mobility_config import (RAW_DATA_FILE, PROCESSED_DATA_FILE, WYOMING_COUNTY_NAMES,
                    WY_COUNTY_FIPS, MONTH_NAMES, YEAR_RANGE)

def safe_parse_json(json_str):
    """Safely parse JSON string"""
    try:
        return json.loads(json_str)
    except:
        return {}

def process_mobility_data():
    """Main processing function - calculates all metrics and saves to Excel"""
    
    # Load data
    try:
        df_full = pd.read_csv(RAW_DATA_FILE)
        print(f" Loaded {len(df_full):,} records")
    except FileNotFoundError:
        print(f" ERROR: File not found!")
        return None
    
    # Parse dates
    if 'DATE_RANGE_START' in df_full.columns:
        df_full['DATE_RANGE_START'] = pd.to_datetime(df_full['DATE_RANGE_START'], errors='coerce')
        df_full['Year'] = df_full['DATE_RANGE_START'].dt.year
        df_full['Month'] = df_full['DATE_RANGE_START'].dt.month
    elif 'Y' in df_full.columns and 'M' in df_full.columns:
        df_full['Year'] = df_full['Y']
        df_full['Month'] = df_full['M']
    else:
        print(" ERROR: No date columns found")
        return None
    
    df_full = df_full[(df_full['Year'] >= YEAR_RANGE[0]) & 
                      (df_full['Year'] <= YEAR_RANGE[1])].copy()
    print(f" Filtered to {YEAR_RANGE[0]}-{YEAR_RANGE[1]}: {len(df_full):,} records")
    
    # Initialize metrics dictionary
    metrics = {name: {} for name in [
        "Total_Inflow", "Internal_Flow", "Inflow_InState", "Inflow_OutState",
        "Outflow_to_WY", "Distinct_Origins_Total", "Distinct_Origins_WY",
        "Distinct_Origins_OutState", "InDegree", "OutDegree", "Weighted_InDegree",
        "Weighted_OutDegree", "Weighted_InStrength", "Weighted_OutStrength",
        "Closeness", "Weighted_Closeness", "Betweenness", "Eigenvector", "PageRank"
    ]}
    
    # Process each time period
    periods = df_full[['Year','Month']].drop_duplicates().sort_values(['Year','Month'])
    total_periods = len(periods)
    
    for idx, (_, row) in enumerate(periods.iterrows(), 1):
        year, month = int(row['Year']), int(row['Month'])
        month_name = MONTH_NAMES[month]
        
        print(f"    [{idx:2d}/{total_periods}] {year}-{month_name:3s}", end=" ... ")
        
        df_month = df_full[(df_full['Year']==year) & (df_full['Month']==month)].copy()
        if len(df_month) == 0:
            print("No data")
            continue
        
        # Parse JSON and expand flows
        df_month["home_dict"] = df_month["DEVICE_HOME_AREAS"].apply(safe_parse_json)
        
        flow_records = []
        for _, r in df_month.iterrows():
            dest_cnty = str(r["AREA"]).strip()[:5]
            for origin_cbg, count in r["home_dict"].items():
                origin_cnty = str(origin_cbg).strip()[:5]
                flow_records.append((origin_cnty, dest_cnty, count))
        
        flows_df = pd.DataFrame(flow_records, columns=["origin_cnty","dest_cnty","count"])
        agg = flows_df.groupby(["origin_cnty","dest_cnty"], as_index=False)["count"].sum()
        agg = agg[agg["dest_cnty"].isin(WY_COUNTY_FIPS)].copy()
        
        snapshot_key = (year, month_name)
        
        # Calculate flow metrics
        no_self = agg["origin_cnty"] != agg["dest_cnty"]
        total_inflow = agg[no_self].groupby("dest_cnty")["count"].sum()
        internal = agg[~no_self].set_index("dest_cnty")["count"]
        inflow_instate = agg[agg["origin_cnty"].isin(WY_COUNTY_FIPS) & no_self].groupby("dest_cnty")["count"].sum()
        inflow_outstate = agg[~agg["origin_cnty"].isin(WY_COUNTY_FIPS)].groupby("dest_cnty")["count"].sum()
        outflow = agg[agg["origin_cnty"].isin(WY_COUNTY_FIPS) & agg["dest_cnty"].isin(WY_COUNTY_FIPS) & no_self].groupby("origin_cnty")["count"].sum()
        distinct_total = agg[no_self].groupby("dest_cnty")["origin_cnty"].nunique()
        distinct_wy = agg[agg["origin_cnty"].isin(WY_COUNTY_FIPS) & no_self].groupby("dest_cnty")["origin_cnty"].nunique()
        distinct_out = agg[~agg["origin_cnty"].isin(WY_COUNTY_FIPS)].groupby("dest_cnty")["origin_cnty"].nunique()
        
        # Build networks
        G = nx.DiGraph()
        G.add_nodes_from(set(agg["origin_cnty"]).union(set(agg["dest_cnty"])))
        for _, r in agg[agg["dest_cnty"].isin(WY_COUNTY_FIPS) & no_self].iterrows():
            G.add_edge(r["origin_cnty"], r["dest_cnty"], weight=r["count"])
        if nx.number_of_selfloops(G) > 0:
            G.remove_edges_from(nx.selfloop_edges(G))
        
        W = nx.Graph()
        W.add_nodes_from(G.nodes())
        for u, v, data in G.edges(data=True):
            w = data.get('weight', 0)
            if W.has_edge(u, v):
                W[u][v]['weight'] += w
            else:
                W.add_edge(u, v, weight=w)
        if nx.number_of_selfloops(W) > 0:
            W.remove_edges_from(nx.selfloop_edges(W))
        
        # Calculate centrality
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        w_in = dict(G.in_degree(weight='weight'))
        w_out = dict(G.out_degree(weight='weight'))
        
        try:
            pr = nx.pagerank(G, alpha=0.85, weight='weight')
        except:
            pr = nx.pagerank(G, alpha=0.85, weight=None)
        
        close = nx.closeness_centrality(W, distance=None)
        
        if W.number_of_edges() > 0:
            try:
                for u, v, data in W.edges(data=True):
                    data['distance'] = (1.0/data['weight']) if data['weight']>0 else float('inf')
                w_close = nx.closeness_centrality(W, distance='distance')
            except:
                w_close = close.copy()
        else:
            w_close = {n: 0.0 for n in W.nodes()}
        
        betw = nx.betweenness_centrality(W, weight=None)
        
        try:
            eigen = nx.eigenvector_centrality_numpy(W, weight='weight', max_iter=1000)
        except:
            eigen = {n: 0.0 for n in W.nodes()}
        
        # Store metrics for WY counties
        for metric in metrics:
            metrics[metric][snapshot_key] = {}
        
        for cnty in WY_COUNTY_FIPS:
            metrics["Total_Inflow"][snapshot_key][cnty] = int(total_inflow.get(cnty, 0))
            metrics["Internal_Flow"][snapshot_key][cnty] = int(internal.get(cnty, 0))
            metrics["Inflow_InState"][snapshot_key][cnty] = int(inflow_instate.get(cnty, 0))
            metrics["Inflow_OutState"][snapshot_key][cnty] = int(inflow_outstate.get(cnty, 0))
            metrics["Outflow_to_WY"][snapshot_key][cnty] = int(outflow.get(cnty, 0))
            metrics["Distinct_Origins_Total"][snapshot_key][cnty] = int(distinct_total.get(cnty, 0))
            metrics["Distinct_Origins_WY"][snapshot_key][cnty] = int(distinct_wy.get(cnty, 0))
            metrics["Distinct_Origins_OutState"][snapshot_key][cnty] = int(distinct_out.get(cnty, 0))
            metrics["InDegree"][snapshot_key][cnty] = int(in_deg.get(cnty, 0))
            metrics["OutDegree"][snapshot_key][cnty] = int(out_deg.get(cnty, 0))
            metrics["Weighted_InDegree"][snapshot_key][cnty] = int(w_in.get(cnty, 0))
            metrics["Weighted_OutDegree"][snapshot_key][cnty] = int(w_out.get(cnty, 0))
            metrics["Weighted_InStrength"][snapshot_key][cnty] = int(w_in.get(cnty, 0))
            metrics["Weighted_OutStrength"][snapshot_key][cnty] = int(w_out.get(cnty, 0))
            metrics["Closeness"][snapshot_key][cnty] = float(close.get(cnty, 0.0))
            metrics["Weighted_Closeness"][snapshot_key][cnty] = float(w_close.get(cnty, 0.0))
            metrics["Betweenness"][snapshot_key][cnty] = float(betw.get(cnty, 0.0))
            metrics["Eigenvector"][snapshot_key][cnty] = float(eigen.get(cnty, 0.0))
            metrics["PageRank"][snapshot_key][cnty] = float(pr.get(cnty, 0.0))
        
        print("✓")
    
    # Convert to DataFrames
    metric_dfs = {}
    
    for metric, data in metrics.items():
        df_metric = pd.DataFrame({
            f"{yr}-{mn}": pd.Series(vals)
            for (yr, mn), vals in data.items()
        })
        
        for cnty in WY_COUNTY_FIPS:
            if cnty not in df_metric.index:
                df_metric.loc[cnty] = 0
        
        df_metric = df_metric.sort_index().fillna(0)
        
        # Sort columns chronologically
        cols = sorted(df_metric.columns, key=lambda x: (
            int(x.split('-')[0]),
            list(MONTH_NAMES.values()).index(x.split('-')[1])
        ))
        df_metric = df_metric[cols]
        
        # Replace FIPS with county names
        df_metric.index = df_metric.index.map(lambda x: WYOMING_COUNTY_NAMES.get(x, x))
        
        metric_dfs[metric] = df_metric
        print(f"    ✓ {metric}")
    
    # Save to Excel
    with pd.ExcelWriter(PROCESSED_DATA_FILE, engine='openpyxl') as writer:
        for metric, df_metric in metric_dfs.items():
            sheet_name = metric if len(metric) <= 31 else metric[:31]
            df_metric.to_excel(writer, sheet_name=sheet_name)
    
    return metric_dfs

if __name__ == "__main__":
    process_mobility_data()