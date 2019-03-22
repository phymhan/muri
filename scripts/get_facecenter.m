src = '/media/ligong/Picasso/Share/cbimfs/Research/Jean/data_1/round1/videos_mp4';
did = dir(fullfile(src, 'Dyad*'));
did = {did.name};

w = 200;

for did_ = did
    vids = dir(fullfile(src, did_{1}, '*.mp4'))';
    for vid_ = vids
        v = VideoReader(fullfile(src, did_{1}, vid_.name));
        fr = readFrame(v);
        imshow(fr)
        fprintf('%s, %s\n', did_{1}, vid_.name)
        % waitforbuttonpress
        [x, y] = ginput(1);
        fid = fopen([fullfile(src, did_{1}, vid_.name) '.txt'], 'w');
        fprintf(fid, '%d %d', round(x), round(y));
        fclose(fid)
    end
end
