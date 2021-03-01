It may be necessary to use the patched version of gdrive:

go get github.com/prasmussen/gdrive
cd $GOPATH/src/github.com/prasmussen/gdrive
git remote add euklid git@github.com:euklid/gdrive.git
git fetch euklid
git checkout exponential-back-off-download

This incorporates exponential back-off to avoid exceeding the rate limit.
