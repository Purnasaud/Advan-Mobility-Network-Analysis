require(WaveletComp)
library(tidyverse)
library(lubridate)
library(zoo)

# PATHS
mov_path <- "F:/Yellow Stone Flood-2022/floodanalysis/Data/Advan Mobility/WY_Daily_County_Stops_2018_2024.csv"
out_dir  <- "F:/Yellow Stone Flood-2022/floodanalysis/Data/Advan Mobility/WY_Wavelet_Movement_2018_2024"
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

# Collect power summaries to build a robust global cap
power_q99 <- c()

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
    # WaveletComp wants a data.frame; include date to enable date labeling
    transmute(date = as.POSIXct(DATE), x_std)
  
  
  # Wavelet parameters (daily data => dt = 1)
  wt <- analyze.wavelet(
    my.data     = series_cty,
    my.series   = "x_std",
    loess.span  = 0,       
    dt          = 1,
    dj          = 1/20,
    lowerPeriod = 2,
    upperPeriod = 360,
    make.pval   = TRUE,
    method      = "white.noise",
    n.sim       = 250       
  )
  
  wavelet_list[[cty]] <- wt
  
  # Power quantile for global color capping
  power_q99 <- c(power_q99, quantile(wt$Power, 0.99, na.rm = TRUE))
}

# Use the median of county 99th-percentile powers 
if (length(power_q99) == 0) stop("No wavelets computed. Check input data.")
global_cap <- median(power_q99, na.rm = TRUE)
message(sprintf("Global color cap (median of county P99): %.3f", global_cap))

# Plot per county with uniform color scale
for (cty in counties) {
  wt <- wavelet_list[[cty]]
  if (is.null(wt)) next
  
  # Cap power to global_cap to unify scale across counties
  wt2 <- wt
  wt2$Power <- pmin(wt$Power, global_cap)
  
  filename <- file.path(out_dir, paste0(gsub("[^A-Za-z0-9_]+", "_", cty), "_wavelet_power.png"))
  png(filename, width = 8, height = 7, units = "in", res = 300)
  
  # Fixed interval color key 
  wt.image(
    wt2,
    my.series          = 1,
    plot.coi           = TRUE,
    plot.contour       = TRUE,
    col.contour        = "white",
    plot.ridge         = TRUE,
    col.ridge          = "black",
    color.key          = "interval",    
    n.levels           = 100,
    color.palette      = "rainbow(100, start=0, end=.7)",
    useRaster          = TRUE,
    max.contour.segments = 250000,
    plot.legend        = TRUE,
    legend.params      = list(
      width = 1.0,
      shrink = 0.9,
      n.ticks = 6,
      label.digits = 2,
      lab = NULL,
      lab.line = 2.5,
      lab.break = paste0("0 - ", format(round(global_cap, 2), nsmall = 2))
    ),
    label.time.axis    = TRUE,
    show.date          = TRUE,
    date.format        = "%Y-%m",
    timelab            = "Time",
    label.period.axis  = TRUE,
    periodlab          = "Period (days)",
    main               = paste("Morlet Wavelet Power Spectrum:", cty),
    lwd                = 2,
    graphics.reset     = TRUE,
    siglvl             = 0.05
  )
  
  dev.off()
  message("Saved: ", filename)
}

message("\nAll counties processed and plotted.")
