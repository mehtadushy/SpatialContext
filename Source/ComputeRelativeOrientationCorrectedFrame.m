function [] = ComputeRelativeOrientationCorrectedFrame(imageFileList, dataBaseDir, params, canSkip)
    %This function computes the relative orientation of keypoint frames by
    %looking at the neighbourhood and adds a correction to the orientation
    %of the frame.
    %It also considers various different contexts and generates new frames
    %based on orientation gleaned from the specified neighbourhoods
  
    if(~exist('params','var'))
      
        params.sigmaFactors = [40;
                               80;
                               ];
        params.completeNeighbourhood = true;
        params.halfNeighbourhood = false;
        params.quarterNeighbourhood = false;
        
    end

    %This defines  the multiplication factors for log(sigma) to define the
    %size of the neighbourhood, 
    if(~isfield(params,'sigmaFactors'))
        params.sigmaFactors = [40;
                               80;
                               ];
    end
    

    if(~isfield(params,'completeNeighbourhood'))
        params.completeNeighbourhood = true;
    end
    if(~isfield(params,'halfNeighbourhood'))
        params.halfNeighbourhood = false;
    end
    if(~isfield(params,'quarterNeighbourhood'))
        params.quarterNeighbourhood = false;
    end
    
    
    %The assumption is that keypoints(with orientation) are already computed 
    fprintf('Compiling Keypoint Frames with Relative Orientation Context \n');
    if(params.completeNeighbourhood)
     fprintf('With Complete Neighbourhood/n');
    end
    if(params.quarterNeighbourhood)
     fprintf('With Quarter Neighbourhood/n');
    end
    if(params.halfNeighbourhood)
     fprintf('With Half Neighbourhood/n');
    end
%%
    
    for f = 1:length(imageFileList)
        
        imageFName = imageFileList{f};
        [dirN base] = fileparts(imageFName);
        baseFName = [dirN filesep base];
        %Load Keypoints for the image
        in_fname = fullfile(dataBaseDir, sprintf('%s%s', baseFName, params.keypointFrameSuffix));
        load(in_fname, 'features');

                      
        if(mod(f,100)==0)
           %sp_progress_bar(pfig,4,4,f,length(imageFileList),'Compiling Sigma Slices:');
           fprintf('f = %d\n', f);
        end
        

        for sf = 1:length(params.sigmaFactors)
            sigmaFactor = params.sigmaFactors(sf);
            outFName = fullfile(dataBaseDir, sprintf('%s_%d_neigh_ROcontext_keypoint_sift_frame.mat', baseFName, sigmaFactor));
            quartOutFName = cell(4,1);
            halfOutFName = cell(4,1);
	    for i = 1:4
               quartOutFName{i,1} = fullfile(dataBaseDir, sprintf('%s_%d_q%d_neigh_ROcontext_keypoint_sift_frame.mat', baseFName, sigmaFactor, i));
	    end
	    for i = 1:4
               halfOutFName{i,1} = fullfile(dataBaseDir, sprintf('%s_%d_h%d_neigh_ROcontext_keypoint_sift_frame.mat', baseFName, sigmaFactor, i));
	    end
        
            if(((size(dir(outFName),1)~=0 && canSkip) || ~(params.completeNeighbourhood)) && ((size(dir(quartOutFName{4,1}),1)~=0 && canSkip) || ~(params.quarterNeighbourhood)) && ((size(dir(halfOutFName{4,1}),1)~=0 && canSkip) || ~(params.halfNeighbourhood)) )
                continue
            end
            newOrientations = zeros(1, length(features.x));
            quartNewOrientations = zeros(4, length(features.x));
            halfNewOrientations = zeros(4, length(features.x));
            
            %for all keypoints
            for k = 1:length(features.x)
                %Compute the mask that defines the neighbourhood for each scale multiple
                %  for all keypoints
                  
                neighbourhoodMask = zeros(size(features.x));
                x = features.x(k);
                y = features.y(k);
                scale = sigmaFactor * log(features.data(3,k));
                if (scale < 1)
                    scale = 1;
                end
%                 x_lo = x - scale;
%                 x_hi = x+scale;
%                 y_lo = y-scale;
%                 y_hi = y+scale;

		%Make it a circular neighbourhood
                %neighbourhoodMask = neighbourhoodMask | ( (features.x > x_lo) & (features.x <= x_hi) & ...
                                            %    (features.y > y_lo) & (features.y <= y_hi));
                neighbourhoodMask = neighbourhoodMask | ( ((features.x-x).^2 + (features.y-y).^2) < scale^2) ;
                neighbourhoodMask(k) = 0;
                
                relevant_frame = features.data(:,neighbourhoodMask == 1);
                ro_correction = 0;
                if(~isempty(relevant_frame) && (params.completeNeighbourhood == true))
                %Try compu
                  %relative_orientations = relevant_frame(4,:) - features.data(4,k);
                  relative_orientations = relevant_frame(4,:);% - features.data(4,k);
                  %Make sure that the relative orientations are from -pi to
                  %pi
                  %relative_orientations(relative_orientations < -pi) = relative_orientations(relative_orientations < -pi) + (2*pi);
                  %relative_orientations(relative_orientations > pi) = relative_orientations(relative_orientations > pi) - (2*pi);
%                  relative_distance2 = (relevant_frame(1,:) - features.data(1,k)).^2 + (relevant_frame(2,:) - features.data(2,k)).^2;
%                  relative_weight = exp(-relative_distance2/ (scale^2));
%                  relative_weight = relative_weight/norm(relative_weight);
                  %try equal weight
                  relative_weight = ones(size(relative_orientations))/length(relative_orientations);
                  ro_correction = dot(relative_orientations, relative_weight);
                end
                %newOrientations(k) =   features.data(4,k) + ro_correction;
                newOrientations(k) =  ro_correction;% - features.data(4,k);
                qro_correction = zeros(1,4);
                hro_correction = zeros(1,4);
                if(~isempty(relevant_frame) && ((params.quarterNeighbourhood == true)||(params.halfNeighbourhood == true)))
                %Now that we have all the interesting keypoints, we would
                %transform the locations and filter the four quadrants and
                %then pool relative orientations in each quadrant
                 relative_locations = [relevant_frame(1,:) - x; relevant_frame(2,:) - y]; %Bring the keypoint to origin
                 rot = features.data(4,k);
                 relative_locations = [cos(rot) sin(rot); -sin(rot) cos(rot)] * relative_locations;
                 quadMask = zeros(4,size(relative_locations,2));
                 quadMask(1,:) = quadMask(1,:) | ((relative_locations(1,:) >= 0 ) & (relative_locations(2,:) >=0 ));
                 quadMask(2,:) = quadMask(2,:) | ( (relative_locations(1,:) <= 0 ) & (relative_locations(2,:) >= 0) );
                 quadMask(3,:) = quadMask(3,:) | ( (relative_locations(1,:) <= 0 ) & (relative_locations(2,:) <= 0) );
                 quadMask(4,:) = quadMask(4,:) | ( (relative_locations(1,:) >= 0 ) & (relative_locations(2,:) <= 0) );
                 
                 for quad = 1:4
                     relative_orientations = relevant_frame(4,quadMask(quad,:)==1) - rot;
%                     Make sure the orientations are from 0 to 2*pi
%                     relative_orientations(relative_orientations < 0) = relative_orientations(relative_orientations<0) + 2*pi;
%                     relative_orientations(relative_orientations >(2*pi)) = relative_orientations(relative_orientations >(2*pi)) - 2*pi;
                    relative_weight = ones(size(relative_orientations))/length(relative_orientations);
                    qro_correction(quad) = dot(relative_orientations, relative_weight);
                 end

                 halfMask = zeros(4,size(relative_locations,2));
                 halfMask(1,:) = quadMask(1,:)|quadMask(2,:);
                 halfMask(2,:) = quadMask(2,:)|quadMask(3,:);
                 halfMask(3,:) = quadMask(3,:)|quadMask(4,:);
                 halfMask(4,:) = quadMask(4,:)|quadMask(1,:);
                 for half = 1:4
                     relative_orientations = relevant_frame(4,halfMask(half,:)==1) - rot;
%                     Make sure the orientations are from 0 to 2*pi
%                     relative_orientations(relative_orientations < 0) = relative_orientations(relative_orientations<0) + 2*pi;
%                     relative_orientations(relative_orientations >(2*pi)) = relative_orientations(relative_orientations >(2*pi)) - 2*pi;
                    relative_weight = ones(size(relative_orientations))/length(relative_orientations);
                    hro_correction(half) = dot(relative_orientations, relative_weight);
                 end
                end

                 for quad = 1:4
                   quartNewOrientations(quad,k) = features.data(4,k) + qro_correction(quad);
                 end
                 for half = 1:4
                   halfNewOrientations(half,k) = features.data(4,k) + hro_correction(half);
                 end
            end

            tempdata = features.data;
            if(params.completeNeighbourhood == true)
                features.data = [features.data(1:3,:); newOrientations];
                save(outFName, 'features');
            end
            if(params.quarterNeighbourhood == true)
                     for quad = 1:4
                        features.data = [features.data(1:3,:); quartNewOrientations(quad,:)];
                        save(quartOutFName{quad}, 'features');
                     end
            end
            if(params.halfNeighbourhood == true)
                     for half = 1:4
                        features.data = [features.data(1:3,:); halfNewOrientations(half,:)];
                        save(halfOutFName{half}, 'features');
                     end
            end
            features.data = tempdata;
            
        end

    end
end

        


