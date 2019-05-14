function [] = ComputeDirectionalRelativeKeypointContext(imageFileList, dataBaseDir, params, canSkip)
    %This function computes an angular spatial count and relative orientation
    %histogram of Keypoint (SIFT) descriptors within the neighbourhood
    %defined by a radius of sigmaFactor * log(keypoint_sigma)
    %The relative spatial histogram bins keypoints within the neighbourhood
    %of a keypoint into 4 quadrants relative to the dominant direction of the keypoint. 
    %The relative orientation histogram similarly bins kepypoints in each sector in the
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
        %Don't keep the count
        %features.directionalContext = zeros(4*length(features.x), 9);
        features.directionalContext = zeros(4*length(features.x), 8);
        features.directionalSift = kron(features.data, ones(4,1));
        features.directionalx = kron(features.x(:), ones(4,1));%repelem(features.x,4);
        features.directionaly = kron(features.y(:), ones(4,1));
                      
        if(mod(f,100)==0)
           %sp_progress_bar(pfig,4,4,f,length(imageFileList),'Compiling Sigma Slices:');
           fprintf('f = %d\n', f);
        end
        
        outFName1 = fullfile(dataBaseDir, sprintf('%s_directional_keypoint_sift.mat', baseFName));

        for sf = 1:length(params.sigmaFactors)
            sigmaFactor = params.sigmaFactors(sf);
            outFName2 = fullfile(dataBaseDir, sprintf('%s_directional_relative_context_neigh%d.mat', baseFName, sigmaFactor));
        
            if(size(dir(outFName2),1)~=0 && canSkip)
                continue
            end
            
            %for all keypoints
            for k = 1:length(features.x)
                %Compute the mask that defines the neighbourhood for each scale multiple
                %  for all keypoints
                features.directionalSift(4*(k-1)+1:4*k,:) = repmat(features.data(k,:),4,1);
                %orientationHist = zeros(1,8);
    
                neighbourhoodMask = zeros(size(features.x));
                x = features.x(k);
                y = features.y(k);
                scale = sigmaFactor * log(features.vl_frame(3,k));
                x_lo = x - scale;
                x_hi = x+scale;
                y_lo = y-scale;
                y_hi = y+scale;

                neighbourhoodMask = neighbourhoodMask | ( (features.x > x_lo) & (features.x <= x_hi) & ...
                                                (features.y > y_lo) & (features.y <= y_hi));
                
                relevant_frame = features.vl_frame(:,neighbourhoodMask == 1);
                
                %Now that we have all the interesting keypoints, we would
                %transform the locations and filter the four quadrants and
                %then pool relative orientations in each quadrant
                relative_locations = [relevant_frame(1,:) - x; relevant_frame(2,:) - y]; %Bring the keypoint to origin
                rot = features.vl_frame(4,k);
                relative_locations = [cos(rot) sin(rot); -sin(rot) cos(rot)] * relative_locations;
                quadMask = zeros(4,size(relative_locations,2));
                quadMask(1,:) = quadMask(1,:) | ((relative_locations(1,:) >= 0 ) & (relative_locations(2,:) >=0 ));
                quadMask(2,:) = quadMask(2,:) | ( (relative_locations(1,:) <= 0 ) & (relative_locations(2,:) >= 0) );
                quadMask(3,:) = quadMask(3,:) | ( (relative_locations(1,:) <= 0 ) & (relative_locations(2,:) <= 0) );
                quadMask(4,:) = quadMask(4,:) | ( (relative_locations(1,:) >= 0 ) & (relative_locations(2,:) <= 0) );
                
                for quad = 1:4
                    relative_orientations = relevant_frame(4,quadMask(quad,:)==1) - features.vl_frame(4,k);
                    %Make sure the orientations are from 0 to 2*pi
                    relative_orientations(relative_orientations < 0) = relative_orientations(relative_orientations<0) + 2*pi;
                    relative_orientations(relative_orientations >(2*pi)) = relative_orientations(relative_orientations >(2*pi)) - 2*pi;
                    orientationHist = hist(relative_orientations, (pi/8:pi/4:2*pi));
                    features.directionalContext(4*(k-1)+quad,:)= orientationHist/sum(quadMask(quad,:),2);
                end
            end
            temp = features;
            clear features;
            features.data = temp.directionalContext;
            features.x = temp.directionalx;
            features.y = temp.directionaly;
            features.wid = temp.wid;
            features.hgt = temp.hgt;
            save(outFName2, 'features');
            %%Utterly inefficient!!!!
            features = temp;
        end
        if(size(dir(outFName1),1)~=0 && canSkip)
               continue
        else
             temp = features;
            clear features;
            features.data = temp.directionalSift;
            features.x = temp.directionalx;
            features.y = temp.directionaly;
            features.wid = temp.wid;
            features.hgt = temp.hgt;
            save(outFName1,'features');
            features=temp;
        end
    end
end

        


