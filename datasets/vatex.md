# Downloading VaTeX 

[Paper](https://arxiv.org/pdf/1904.03493.pdf)\
[Project Page Download](https://eric-xw.github.io/vatex-website/download.html): Contains the video ids and captions data.

1. We use this [repo](https://github.com/cvdfoundation/kinetics-dataset/tree/main) to download kinetic-600 dataset, source of VaTeX dataset.

2. We just need the validation dataset videos for VaTeX dataset. First run this command to download data:
```
#!/bin/bash

root_dl="k600"
root_dl_targz="k600_targz"

# Make root directories
[ ! -d $root_dl ] && mkdir $root_dl
[ ! -d $root_dl_targz ] && mkdir $root_dl_targz


# Download validation tars, will resume
curr_dl=${root_dl_targz}/val
url=https://s3.amazonaws.com/kinetics/600/val/k600_val_path.txt
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c -i $url -P $curr_dl

# Download test tars, will resume
# Download annotations csv files
curr_dl=${root_dl}/annotations
url_v=https://s3.amazonaws.com/kinetics/600/annotations/val.txt
wget -c $url_v -P $curr_dl

# Download readme
url=http://s3.amazonaws.com/kinetics/600/readme.md
wget -c $url -P $root_dl

# Downloads complete
echo -e "\nDownloads complete! Now run extractor, k600_extractor.sh"
```

3. Then, run this command to extract the downloaded data:

```
#!/bin/bash

# Download directories vars
root_dl="k600"
root_dl_targz="k600_targz"

# Make root directories
[ ! -d $root_dl_targz ] && echo -e "\nRun k600_downloaders.sh"
[ ! -d $root_dl ] && mkdir $root_dl


# Extract validation
curr_dl=$root_dl_targz/val
curr_extract=$root_dl/val
[ ! -d $curr_extract ] && mkdir -p $curr_extract
find $curr_dl -type f | while read file; do mv "$file" `echo $file | tr ' ' '_'`; done
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done

# Extraction complete
echo -e "\nExtractions complete!"
```