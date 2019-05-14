function [] = ComputeCoarseNeighbourhoodHistogram( imageFileList, dataBaseDir, textonSuffix, params, canSkip)
    %This function is meant to compile histograms of a descriptor
    %(textonSuffix) in the neighbourhood of keypoints (passed implicitly
    %via keyPointFeatureSuffix)
    if(~exist('params','var'))
        params.dictionarySize = 20;
        params.sigmaFactor =  80;
        params.scaleWeighted =  false;
        params.keypointFeatureSuffix = '_keypoint_sift.mat';
    end

    %This defines  the multiplication factors for log(sigma) to define the
    %size of the neighbourhood, 
    if(~isfield(params,'sigmaFactor'))
        params.sigmaFactor = 80;
    end
    if(~isfield(params,'dictionarySize'))
        params.dictionarySize = 20;
    end
    if(~isfield(params,'scaleWeighted'))
        params.scaleWeighted = false;
    end
    if(~isfield(params,'keypointFeatureSuffix'))
        params.keypointFeatureSuffix = '_keypoint_sift.mat';
    end
    
    %This assumes that the features are already densely computed for the
    %images, a vocabulary built, and features assigned to the corresponding
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
        %Load coarse sift texton indices
        in_fname = fullfile(dataBaseDir, sprintf('%s%s', baseFName, textonSuffix));
        load(in_fname, 'texton_ind');
                        
        if(mod(f,100)==0)
           %sp_progress_bar(pfig,4,4,f,length(imageFileList),'Compiling Sigma Slices:');
           fprintf('f = %d\n', f);
        end
        
            sigmaFactor = params.sigmaFactor;
            outFName = fullfile(dataBaseDir, sprintf('%s_%d_neigh_contexthist%d.mat', baseFName, sigmaFactor, params.dictionarySize));
        
            if(size(dir(outFName),1)~=0 && canSkip)
                continue
            end
            
            %for all keypoints
            for k = 1:length(features.x)
                %features.data(k,:) = histogram
                %Compute the mask that defines the neighbourhood for each scale multiple
                %  for all keypoints
                sf_hist = zeros(1, params.dictionarySize);
                   scale = sigmaFactor;
                   neighbourhoodMask = zeros(size(texton_ind.data(:,1)));
                   x = features.x(k);
                   y = features.y(k);
                   scale = scale * log(keypt_features.vl_frame(3,k));
		   if(scale <1)
		     scale =1;
		   end
                    x_lo = x - scale;
                    x_hi = x+scale;
                    y_lo = y-scale;
                    y_hi = y+scale;

                    neighbourhoodMask = neighbourhoodMask | ( (texton_ind.x > x_lo) & (texton_ind.x <= x_hi) & ...
                                                (texton_ind.y > y_lo) & (texton_ind.y <= y_hi));
                
                    texton_region = texton_ind.data(neighbourhoodMask == 1,1);
		    neigh_scales = keypt_features.vl_frame(3, neighbourhoodMask ==1);
		    if(~params.scaleWeighted)
			 sf_hist =  hist(texton_region, 1:params.dictionarySize);
			 features.data(k,:) = sf_hist/length(texton_region);
			 features.data(k,:) = sf_hist;
		    else
			sf_hist = zeros(1, params.dictionarySize);
		    	for idx = 1:length(texton_region)
		            sf_hist(texton_region(idx)) = sf_hist(texton_region(idx))	+ neigh_scales(idx);
		        end
			features.data(k,:) = sqrt(sf_hist)/sum(sqrt(sf_hist),2);
		    end
		    
            end
            
            save(outFName, 'features');
    end
end

        


