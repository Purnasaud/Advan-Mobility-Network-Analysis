require(WaveletComp)
library(tidyverse)
library(lubridate)
library(zoo)

# PATHS
mov_path <- "F:/Yellow Stone Flood-2022/floodanalysis/Data/Advan Mobility/WY_Daily_County_Stops_2018_2024.csv"
out_dir  <- "F:/Yellow Stone Flood-2022/floodanalysis/Data/Advan Mobility/WY_Wavelet_Movement_2018_2024/Average Wavelet Power Spectrum"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Load datasets
county_movements <- read_csv(mov_path, show_col_types = FALSE) %>%
  mutate(
    DATE = as.Date(DATE),
    COUNTY_NAME = as.character(COUNTY_NAME)
  )

# Limit to desired window 
date_min <- as.Date("2018-01-01")
date_max <- as.Date("2024-12-31")
county_movements <- county_movements %>% filter(DATE >= date_min, DATE <= date_max)

# Build regular daily series per county
mk_daily <- function(df_cty) {
  full_dates <- tibble(DATE = seq(date_min, date_max, by = "day"))
  df_cty %>%
    select(DATE, TOTAL_DAILY_STOPS) %>%
    right_join(full_dates, by = "DATE") %>%
    mutate(TOTAL_DAILY_STOPS = coalesce(TOTAL_DAILY_STOPS, 0))
}

# Prepare standardized series for each county & compute wavelets
counties <- sort(unique(county_movements$COUNTY_NAME))

wavelet_list <- vector("list", length(counties))
names(wavelet_list) <- counties

message("Computing wavelets for ", length(counties), " counties...")

for (cty in counties) {
  message("  • ", cty)
  series_cty <- county_movements %>%
    filter(COUNTY_NAME == cty) %>%
    mk_daily() %>%
    arrange(DATE)
  
  series_cty <- series_cty %>% mutate(x = log1p(TOTAL_DAILY_STOPS))
  
  # Standardize (mean 0, sd 1) for wavelet
  series_cty <- series_cty %>%
    mutate(x_std = as.numeric(scale(x))) %>%
    transmute(date = as.POSIXct(DATE), x_std)
  
  # Wavelet parameters with extended period range to 500 days
  wt <- analyze.wavelet(
    my.data     = series_cty,
    my.series   = "x_std",
    loess.span  = 0,       
    dt          = 1,
    dj          = 1/20,
    lowerPeriod = 2,
    upperPeriod = 500,      
    make.pval   = TRUE,
    method      = "white.noise",
    n.sim       = 250       
  )
  
  wavelet_list[[cty]] <- wt
}

message("\nAll wavelets computed.")

# ============================================================================
# PLOT AVERAGE WAVELET POWER SPECTRUM FOR EACH COUNTY
# ============================================================================

message("\nPlotting average wavelet spectra...")

for (cty in counties) {
  wt <- wavelet_list[[cty]]
  if (is.null(wt)) next
  
  message("  • Plotting average spectrum for ", cty)
  
  # Create filename for average spectrum plot
  filename_avg <- file.path(out_dir, paste0(gsub("[^A-Za-z0-9_]+", "_", cty), "_wavelet_avg_spectrum.png"))
  
  # Plot the average spectrum
  png(filename_avg, width = 10, height = 6, units = "in", res = 300)
  
  wt.avg(
    wt,
    my.series = 1,
    show.siglvl = TRUE,
    siglvl = 0.05,
    sigcol = "red",
    sigpch = 20,
    sigcex = 0.5,
    main = paste("Average Wavelet Power Spectrum:", cty),
    periodlab = "Period (days)",
    averagelab = "Average Power",
    show.legend = TRUE,
    legend.coords = "topright",
    lwd = 2
  )
  
  dev.off()
  message("    Saved: ", filename_avg)
}

message("\nAll average spectra plotted.")