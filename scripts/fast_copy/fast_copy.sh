#!/bin/bash


# Copy DEM tar.gz data from gws to pw3
mkdir -p /work/scratch-pw3/mespi/majorTOM/world/Core-DEM-tar-gz
find /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-DEM-tar-gz -maxdepth 1 -name '*.tar.gz' \
  | xargs -P 32 -I{} cp -v {} /work/scratch-pw3/mespi/majorTOM/world/Core-DEM-tar-gz/


# Copy S1RTC tar.gz data from gws to pw3
mkdir -p /work/scratch-pw3/mespi/majorTOM/world/Core-S1RTC-tar-gz
find /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-S1RTC-tar-gz -maxdepth 1 -name '*.tar.gz' \
  | xargs -P 32 -I{} cp -v {} /work/scratch-pw3/mespi/majorTOM/world/Core-S1RTC-tar-gz/


