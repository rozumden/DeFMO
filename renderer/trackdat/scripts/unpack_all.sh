#!/bin/bash

dl="${1:-./dl}"
data="${2:-./data}"
scripts="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

set -x

bash "${scripts}/unpack_otb.sh" "${dl}/otb" "${data}/otb"
bash "${scripts}/unpack_vot.sh" "${dl}/vot2013" "${data}/vot2013"
bash "${scripts}/unpack_vot.sh" "${dl}/vot2014" "${data}/vot2014"
bash "${scripts}/unpack_vot.sh" "${dl}/vot2015" "${data}/vot2015"
bash "${scripts}/unpack_vot.sh" "${dl}/vot2016" "${data}/vot2016"
bash "${scripts}/unpack_vot.sh" "${dl}/vot2017" "${data}/vot2017"
bash "${scripts}/unpack_vot.sh" "${dl}/vot2018" "${data}/vot2018"
bash "${scripts}/unpack_vot.sh" "${dl}/vot2018_longterm" "${data}/vot2018_longterm"
bash "${scripts}/unpack_tc128.sh" "${dl}/tc128" "${data}/tc128"
bash "${scripts}/unpack_alov.sh" "${dl}/alov" "${data}/alov"
bash "${scripts}/unpack_nuspro.sh" "${dl}/nuspro" "${data}/nuspro"
bash "${scripts}/unpack_uav123.sh" "${dl}/uav123" "${data}/uav123"
bash "${scripts}/unpack_dtb70.sh" "${dl}/dtb70" "${data}/dtb70"
bash "${scripts}/unpack_tlp.sh" "${dl}/tlp" "${data}/tlp"
bash "${scripts}/unpack_nfs.sh" "${dl}/nfs" "${data}/nfs"
bash "${scripts}/unpack_ilsvrc.sh" "${dl}/ilsvrc" "${data}/ilsvrc"
TRACKINGNET_RATE=1 bash "${scripts}/unpack_trackingnet.sh" "${dl}/trackingnet" "${data}/trackingnet"
TRACKINGNET_RATE=10 bash "${scripts}/unpack_trackingnet.sh" "${dl}/trackingnet" "${data}/trackingnet_10"
