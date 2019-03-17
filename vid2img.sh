 for f in video/*; do
     out_dir=${f/video/video_2}
     mkdir $out_dir
     ffmpeg -i $f $out_dir/img%04d.jpg
 done
