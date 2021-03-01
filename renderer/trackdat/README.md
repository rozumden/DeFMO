# trackdat

## Setup datasets

We show how to set up the VOT 2018 dataset as an example.

To download the compressed dataset:
```bash
VOT_YEAR=2018 bash scripts/download_vot.sh dl/vot2018
```
To unpack the compressed dataset to a temporary directory and then save it as a tarball:
```bash
# Unpack from dl/vot2018 to temp.
temp=$(mktemp -d)
bash scripts/unpack_vot.sh dl/vot2018 $temp/vot2018
# Pack from temp to tar/original/vot2018.tar
mkdir -p tar/original
tar -cf tar/original/vot2018.tar -C $temp vot2018
rm -r $temp
```
