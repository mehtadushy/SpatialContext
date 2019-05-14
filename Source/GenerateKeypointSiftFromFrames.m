function [ ] = GenerateKeypointSiftFromFrames(imageFileList, imageBaseDir, dataBaseDir, params, canSkip, pfig)
    if(~exist('params','var'))
        params.maxImageSize = 1000
        params.gridSpacing = 8
        params.patchSize = 16
        params.dictionarySize = 200
        params.numTextonImages = 50
        params.pyramidLevels = 3
        params.oldSift = false;
        params.SIFTMagnif = 3;
        params.keypointFramePrefix = '';

    end

    if(~isfield(params,'maxImageSize'))
        params.maxImageSize = 1000;  
    end
    if(~isfield(params,'SIFTMagnif'))
        params.SIFTMagnif = 3;  
    end
    if(~exist('canSkip','var'))
     canSkip = 1;
    end
    if(exist('pfig','var'))
        tic;
    end
    if(~isfield(params,'keypointFramePrefix'))
        params.keypointFramePrefix = '';
    end
    
    fprintf('Computing Non-Dense Sift Descriptors \n\n');
%%
    for f = 1:length(imageFileList)
        imageFName = imageFileList{f};
        [dirN base] = fileparts(imageFName);
        baseFName = [dirN filesep base];
        inFName = fullfile(dataBaseDir, sprintf('%s%s_keypoint_sift_frame.mat', baseFName, params.keypointFramePrefix));
        outFName = fullfile(dataBaseDir, sprintf('%s%s_keypoint_sift.mat', baseFName, params.keypointFramePrefix));
        imageFName = fullfile(imageBaseDir, imageFName);
        
        if(mod(f,100)==0 && exist('pfig','var'))
            %sp_progress_bar(pfig,1,4,f,length(imageFileList));
        end
        if(exist(outFName,'file')~=0 && canSkip)
            %fprintf('Skipping %s\n', imageFName);
            continue;
        end
        
        I = sp_load_image(imageFName);
        load(inFName, 'features');
        
        [frame, desc] = vl_sift(single(I), 'magnif', params.SIFTMagnif, 'frames', features.data);
        
        features.data = sp_normalize_sift(double(desc'));
        features.vl_frame = frame;

        if(exist('pfig', 'var'))
         %sp_progress_bar(pfig,1,4,f,length(imageFileList),'Generating Sift From Frames:');
        end
        sp_make_dir(outFName);
        save(outFName, 'features');
    end
    


end