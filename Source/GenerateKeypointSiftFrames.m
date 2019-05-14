function [ ] = GenerateKeypointSiftFrames(imageFileList, imageBaseDir, dataBaseDir, params, canSkip, pfig)
    if(~exist('params','var'))
        params.maxImageSize = 1000
        params.gridSpacing = 8
        params.patchSize = 16
        params.dictionarySize = 200
        params.numTextonImages = 50
        params.pyramidLevels = 3
        params.oldSift = false;
        params.SIFTMagnif = 3;
        params.axisAlignedSIFT = false;
    end

    if(~isfield(params,'maxImageSize'))
        params.maxImageSize = 1000;  
    end
    if(~isfield(params,'SIFTMagnif'))
        params.SIFTMagnif = 3;  
    end
    if(~isfield(params,'axisAlignedSIFT'))
        params.axisAlignedSIFT = false;  
    end
    if(~exist('canSkip','var'))
     canSkip = 1;
    end
    if(exist('pfig','var'))
        tic;
    end
    fprintf('Computing Non-Dense Sift Frames \n\n');
%%
    for f = 1:length(imageFileList)
        imageFName = imageFileList{f};
        [dirN base] = fileparts(imageFName);
        baseFName = [dirN filesep base];
        outFName = fullfile(dataBaseDir, sprintf('%s_keypoint_sift_frame.mat', baseFName));
        imageFName = fullfile(imageBaseDir, imageFName);
        
        if(mod(f,100)==0 && exist('pfig','var'))
            %sp_progress_bar(pfig,1,4,f,length(imageFileList));
        end
        if(exist(outFName,'file')~=0 && canSkip)
            %fprintf('Skipping %s\n', imageFName);
            continue;
        end
        
        I = sp_load_image(imageFName);
        [hgt wid] = size(I);
        if min(hgt,wid) > params.maxImageSize
            I = imresize(I, params.maxImageSize/min(hgt,wid), 'bicubic');
            fprintf('Loaded %s: original size %d x %d, resizing to %d x %d\n', ...
            imageFName, wid, hgt, size(I,2), size(I,1));
            [hgt wid] = size(I);
        end
        if(~params.axisAlignedSIFT)
            %Make sure that the number of keypoints used in both cases are
            %the same
            %frame = vl_covdet(single(I));
            frame = vl_sift(single(I), 'magnif', params.SIFTMagnif);
        else
            frame = vl_sift(single(I), 'magnif', params.SIFTMagnif);
            frame = [frame(1:3,:); zeros(1,size(frame,2))];
        end
        features.wid = wid;
        features.hgt = hgt;
        features.data = frame;
        features.x = frame(1,:)';
        features.y = frame(2,:)';
        %sp_progress_bar(pfig,1,4,f,length(imageFileList),'Generating Sift Frames:');
        sp_make_dir(outFName);
        save(outFName, 'features');
    end
    


end
