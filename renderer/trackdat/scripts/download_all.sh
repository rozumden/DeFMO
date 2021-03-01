#!/bin/bash

dl="${1:-./dl}"
scripts="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

set -x

bash "${scripts}/download_otb.sh" "${dl}/otb"
VOT_YEAR=2013 bash "${scripts}/download_vot.sh" "${dl}/vot2013"
VOT_YEAR=2014 bash "${scripts}/download_vot.sh" "${dl}/vot2014"
VOT_YEAR=2015 bash "${scripts}/download_vot.sh" "${dl}/vot2015"
VOT_YEAR=2016 bash "${scripts}/download_vot.sh" "${dl}/vot2016"
VOT_YEAR=2017 bash "${scripts}/download_vot.sh" "${dl}/vot2017"
VOT_YEAR=2018 bash "${scripts}/download_vot.sh" "${dl}/vot2018"
VOT_YEAR=2018 VOT_CHALLENGE=longterm bash "${scripts}/download_vot.sh" "${dl}/vot2018"
bash "${scripts}/download_tc128.sh" "${dl}/tc128"
bash "${scripts}/download_alov.sh" "${dl}/alov"
bash "${scripts}/download_nuspro.sh" "${dl}/nuspro"
bash "${scripts}/download_uav123.sh" "${dl}/uav123"
bash "${scripts}/download_dtb70.sh" "${dl}/dtb70"
bash "${scripts}/download_tlp.sh" "${dl}/tlp"
bash "${scripts}/download_nfs.sh" "${dl}/nfs"
bash "${scripts}/download_ilsvrc.sh" "${dl}/ilsvrc"
bash "${scripts}/download_trackingnet.sh" "${dl}/trackingnet"
