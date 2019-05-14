function [features] = sp_gen_sift(imageFName,params)
    I = sp_load_image(imageFName);

    [hgt wid] = size(I);
    if min(hgt,wid) > params.maxImageSize
        I = imresize(I, params.maxImageSize/min(hgt,wid), 'bicubic');
        fprintf('Loaded %s: original size %d x %d, resizing to %d x %d\n', ...
            imageFName, wid, hgt, size(I,2), size(I,1));
        [hgt wid] = size(I);
    end


        [siftArr, gridX, gridY] = sp_dense_sift(I, params.gridSpacing, params.patchSize);
        siftArr = reshape(siftArr,[size(siftArr,1)*size(siftArr,2) size(siftArr,3)]);
        %siftArr has sift features in each row, with locations of features
        %given by linear indexing into gridX and gridY

    features.data = siftArr;
    features.x = gridX(:);% + params.patchSize/2 - 0.5;
    features.y = gridY(:);% + params.patchSize/2 - 0.5;
    features.wid = wid;
    features.hgt = hgt;
end