function [] = ComputeRelativeKeypointContext(imageFileList, dataBaseDir, params, canSkip)
    %This function computes an angular spatial and relative orientation
    %histogram of Keypoint (SIFT) descriptors within the neighbourhood
    %defined by a radius of sigmaFactor * log(keypoint_sigma)
    %The relative spatial histogram bins keypoints within the neighbourhood
    %of a keypoint into 4 quadrants relative to the dominant direction of the keypoint. 
    %The relative orientation histogram similarly bins kepypoints in the
    %neighbourhood of a keypoint into 8 relative orientation bins
    if(~exist('params','var'))
      
        params.sigmaFactors = [20;
                               25;
                               30;
                               40;
                               ];
        params.keypointFeatureSuffix = '_keypoint_sift.mat';
    end

    %This defines  the multiplication factors for log(sigma) to define the
    %size of the neighbourhood, 
    if(~isfield(params,'sigmaFactors'))
        params.sigmaFactors = [20;
                               25;
                               30;
                               40;
                               ];
    end
    

    if(~isfield(params,'keypointFeatureSuffix'))
        params.keypointFeatureSuffix = '_keypoint_sift.mat';
    end
    
    %The assumption is that keypoints(with orientation) are already computed 
    fprintf('Compiling Relative Keypoint Context\n');
%%
    
    for f = 1:length(imageFileList)
        
        imageFName = imageFileList{f};
        [dirN base] = fileparts(imageFName);
        baseFName = [dirN filesep base];
        %Load Keypoints for the image
        in_fname = fullfile(dataBaseDir, sprintf('%s%s', baseFName, params.keypointFeatureSuffix));
        load(in_fname, 'features');
        features.spatialHist = zeros(length(features.x), 4);
        features.orientationHist = zeros(length(features.x), 8);
                      
        if(mod(f,100)==0)
           %sp_progress_bar(pfig,4,4,f,length(imageFileList),'Compiling Sigma Slices:');
           fprintf('f = %d\n', f);
        end
        
        for sf = 1:length(params.sigmaFactors)
            sigmaFactor = params.sigmaFactors(sf);
            outFName = fullfile(dataBaseDir, sprintf('%s_keypoint_sift_relative_context_neigh%d.mat', baseFName, sigmaFactor));
        
            if(size(dir(outFName),1)~=0 && canSkip)
                continue
            end
            
            %for all keypoints
            for k = 1:length(features.x)
                %Compute the mask that defines the neighbourhood for each scale multiple
                %  for all keypoints
                spatialHist = zeros(1,4);
                %orientationHist = zeros(1,8);
    
                neighbourhoodMask = zeros(size(features.x));
                x = features.x(k);
                y = features.y(k);
                scale = sigmaFactor * log(features.vl_frame(3,k));
                if (scale < 1)
                    scale = 1;
                end
                x_lo = x - scale;
                x_hi = x+scale;
                y_lo = y-scale;
                y_hi = y+scale;

                neighbourhoodMask = neighbourhoodMask | ( (features.x > x_lo) & (features.x <= x_hi) & ...
                                                (features.y > y_lo) & (features.y <= y_hi));
                
                relevant_region = features.vl_frame(:,neighbourhoodMask == 1);
                region_size = size(relevant_region,2);
                
                relative_orientations = relevant_region(4,:) - features.vl_frame(4,k);
                %Make sure the orientations are from 0 to 2*pi
                relative_orientations(relative_orientations < 0) = relative_orientations(relative_orientations<0) + 2*pi;
                relative_orientations(relative_orientations >(2*pi)) = relative_orientations(relative_orientations >(2*pi)) - 2*pi;
                
                relative_locations = [relevant_region(1,:) - x; relevant_region(2,:) - y]; %Bring the keypoint to origin
                rot = features.vl_frame(4,k);
                relative_locations = [cos(rot) sin(rot); -sin(rot) cos(rot)] * relative_locations;
                
                spatialHist(1) = sum( (relative_locations(1,:) > 0 ) & (relative_locations(2,:) > 0 ) );
                spatialHist(2) = sum( (relative_locations(1,:) < 0 ) & (relative_locations(2,:) > 0 ) );
                spatialHist(3) = sum( (relative_locations(1,:) < 0 ) & (relative_locations(2,:) < 0 ) );
                spatialHist(4) = sum( (relative_locations(1,:) > 0 ) & (relative_locations(2,:) < 0 ) );
                
                orientationHist = hist(relative_orientations, (pi/8:pi/4:2*pi));
                
                features.spatialHist(k,:) = spatialHist/region_size;
                features.orientationHist(k,:) = orientationHist/region_size;
            end
            
            save(outFName, 'features');
        end
    end
end

        


