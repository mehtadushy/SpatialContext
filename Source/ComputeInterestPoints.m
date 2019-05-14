function [ interest_points ] = ComputeInterestPoints(imageFileList, imageBaseDir, dataBaseDir, params, numPoints, canSkip)
    if(~exist('params','var'))
        params.maxImageSize = 1000
        params.gridSpacing = 8
        params.patchSize = 16
        params.dictionarySize = 200
        params.numTextonImages = 50
        params.pyramidLevels = 3
        params.oldSift = false;
        params.keypointDetector = 'DoG';
    end

    if(~isfield(params,'maxImageSize'))
        params.maxImageSize = 1000;  %The only param we really care about here
    end
    if(~isfield(params,'keypointDetector'))
        params.keypointDetector = 'DoG';
    end
    if(~exist('canSkip','var'))
     canSkip = 1;
    end
    if(~exist('numPoints','var'))
     numPoints = 100;
    end   
    fprintf('Computing %d Largest Interest Points \n\n', numPoints);
    
    outFName = fullfile(dataBaseDir, sprintf('interest_points_%d_%s_all.mat',numPoints,params.keypointDetector));
    
    interest_points = cell(length(imageFileList),1);

    if(size(dir(outFName),1)~=0 && canSkip)
        load(outFName, 'interest_points');
        fprintf('Found file %s. Skipping computation of interest points',outFName);
    else
        %Iterate over all images and make the interest points
        for f = 1:length(imageFileList)
            fname = fullfile(imageBaseDir, imageFileList{f});
            I = sp_load_image(fname);
            %%
            %If images were resized when computing SIFT, we should resize
            %them here as well, so that we can reliably extract the
            %neighbourhood of the interest points in the sift sampling grid
            [hgt wid] = size(I);
            if min(hgt,wid) > params.maxImageSize
                I = imresize(I, params.maxImageSize/min(hgt,wid), 'bicubic');
                fprintf('Loaded %s: original size %d x %d, resizing to %d x %d\n', ...
                    imageFName, wid, hgt, size(I,2), size(I,1));
                
            end
            %%
                % Now that it is assured that the locations we get for
                % interest points correspond to the same coordinate system
                % as the sift descriptors, go ahead and compute the
                % interest points
                feat = vl_covdet(single(I), 'Method', params.keypointDetector);
                %sort the features in asciending order of scale
                [~, I] = sort(feat(3,:));
                feat = feat(:,I);
                if (size(feat,2)<numPoints)
                    lower_index = 1;
                else
                    lower_index = size(feat,2)-numPoints+1;
                end 
%                 ip.x = feat(1,(size(feat,2):-1:lower_index));
%                 ip.y = feat(2,(size(feat,2):-1:lower_index));
%                 ip.scale = feat(3,(size(feat,2):-1:lower_index));
                interest_points{f} = feat(:,(size(feat,2):-1:lower_index));
        end
        save(outFName, 'interest_points');
    end
    


end