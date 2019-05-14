function [] = ComputeNeighborhoodPoolingHistogram( imageFileList, dataBaseDir, textonSuffix, params, canSkip)
    %This function is meant to compile histograms of a descriptor
    %(textonSuffix) in the neighbourhood of keypoints (passed implicitly
    %via keyPointFeatureSuffix)
    if(~exist('params','var'))
        params.dictionarySize = 200;
       
        params.sigmaFactors = [6 10;
                               6 14;
                               8 16;
                               10 16;
                               10 18];
        params.keypointFeatureSuffix = '_keypoint_sift.mat';
    end

    %This defines  the multiplication factors for sigma to define the
    %size of the neighbourhood, 
    if(~isfield(params,'sigmaFactors'))
        params.sigmaFactors = [6 10;
                               6 14;
                               8 16;
                               10 16;
                               10 18];
    end
      if(~isfield(params,'dictionarySize'))
        params.dictionarySize = 200;
    end
    if(~isfield(params,'keypointFeatureSuffix'))
        params.keypointFeatureSuffix = '_keypoint_sift.mat';
    end
    
    %This assumes that the features are already densely computed for the
    %images, a vocabulary built, and features assigned to the correcponding
    %textons
    fprintf('Compiling Neighbourhood Pooling Histograms\n');
%%
    
    for f = 1:length(imageFileList)
        
        imageFName = imageFileList{f};
        [dirN base] = fileparts(imageFName);
        baseFName = [dirN filesep base];
        %Load Keypoints for the image
        in_fname = fullfile(dataBaseDir, sprintf('%s%s', baseFName, params.keypointFeatureSuffix));
        load(in_fname, 'features');
        keypt_features = features;
        clear features;
        features.wid = keypt_features.wid;
        features.hgt = keypt_features.hgt;
        features.x = keypt_features.x;
        features.y = keypt_features.y;
        features.data = zeros(length(features.x), params.dictionarySize);
        %Load dense sift texton indices
        in_fname = fullfile(dataBaseDir, sprintf('%s%s', baseFName, textonSuffix));
        load(in_fname, 'texton_ind');
                        
        if(mod(f,100)==0)
           %sp_progress_bar(pfig,4,4,f,length(imageFileList),'Compiling Sigma Slices:');
           fprintf('f = %d\n', f);
        end
        
        for sf = 1:size(params.sigmaFactors,1)
            sigmaFactor = params.sigmaFactors(sf,:);
            outFName = fullfile(dataBaseDir, sprintf('%s_%d_%d_neighborhood_histsize%d.mat', baseFName, sigmaFactor(1), sigmaFactor(2), params.dictionarySize));
        
            if(size(dir(outFName),1)~=0 && canSkip)
                continue
            end
            
            %for all keypoints
            for k = 1:length(features.x)
                %features.data(k,:) = histogram
                %Compute the mask that defines the neighbourhood for each scale multiple
                %  for all keypoints
                sf_hist = zeros(2, params.dictionarySize);
                sf_size = [0 0];
                
                for s = 1:2
                   scale = sigmaFactor(s);
                   neighbourhoodMask = zeros(size(texton_ind.data));
                   x = features.x(k);
                   y = features.y(k);
                   scale = scale * keypt_features.vl_frame(3,k);
                    x_lo = x - scale;
                    x_hi = x+scale;
                    y_lo = y-scale;
                    y_hi = y+scale;

                    neighbourhoodMask = neighbourhoodMask | ( (texton_ind.x > x_lo) & (texton_ind.x <= x_hi) & ...
                                                (texton_ind.y > y_lo) & (texton_ind.y <= y_hi));
                
                    texton_region = texton_ind.data(neighbourhoodMask == 1);
                    sf_size(s) = length(texton_region);
                    sf_hist(s,:) =  hist(texton_region, 1:params.dictionarySize);
                end
                features.data(k,:) = (sf_hist(2,:) - sf_hist(1,:));
                if ((sf_size(2) - sf_size(1)) > 0)
                    features.data(k,:) = features.data(k,:) / (sf_size(2) - sf_size(1));
                end
            end
            
            save(outFName, 'features');
        end
    end
end

        


