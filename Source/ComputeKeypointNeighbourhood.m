function [] = ComputeKeypointNeighbourhood(imageFileList, dataBaseDir, params, canSkip)
    %This function computes the keypoints that occur in the neighbourhood of each keypoint
    % and logs the relative orientaion information as well as a coarse histogram of keypoints.
    %This is done for various different sorts of neighbourhoods
    if(~exist('params','var'))
      
        params.sigmaFactors = [40;
                               80;
                               ];
	%This carries vl_frame which carries orientation and scale information	       
	%Also, the soft_histogram filename is automatically constructed from
	%the feature suffix name
        params.keypointFeatureSuffix = '_keypoint_sift.mat';
        params.keypointHistogramSize = [20;
                               40;
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
    
    if(~isfield(params,'keypointHistogramSize'))
        params.keypointHistogramSize = [20;
                               40;
                               ];
    end

    if(~isfield(params,'keypointFeatureSuffix'))
        params.keypointFeatureSuffix = '_keypoint_sift.mat';
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
    fprintf('Compiling Keypoint Context with Relative Orientation and SIFT Histograms \n');
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
        [dirN, base] = fileparts(imageFName);
        baseFName = [dirN filesep base];
        %Load Keypoints for the image
        in_fname = fullfile(dataBaseDir, sprintf('%s%s', baseFName, params.keypointFeatureSuffix));
        imFeat = load(in_fname, 'features');
	
        features = imFeat.features;
        clear features.data features.vl_sift features.vl_frame;

                      
        if(mod(f,100)==0)
           %sp_progress_bar(pfig,4,4,f,length(imageFileList),'Compiling Sigma Slices:');
           fprintf('f = %d\n', f);
        end
        

        for sf = 1:length(params.sigmaFactors)
            sigmaFactor = params.sigmaFactors(sf);

            neighMask = zeros(length(features.x));
            quartNeighMask = zeros(4, length(features.x), length(features.x));
            halfNeighMask = zeros(4, length(features.x), length(features.x));
            
            relOrientationHist = zeros(length(features.x), 8);
            quartRelOrientationHist = zeros(4, length(features.x), 8);
            halfRelOrientationHist = zeros(4, length(features.x), 8);

            %Go in and compute the neighbourhood information for whole, quarter and half neighbourhoods.
            %Also compute and log the relative orientation information while at it.
            
            for k = 1:length(features.x)
                %Compute the mask that defines the neighbourhood for each scale multiple
                %  for all keypoints
                  
                neighbourhoodMask = zeros(size(features.x));
                x = features.x(k);
                y = features.y(k);
                scale = sigmaFactor * log(imFeat.features.vl_frame(3,k));
                if (scale < 1)
                    scale = 1;
                end
                neighbourhoodMask = neighbourhoodMask | ( ((features.x-x).^2 + (features.y-y).^2) < scale^2) ;
                neighbourhoodMask(k) = 0;
                neighMask(k,:) = neighbourhoodMask; 
                
                relevant_frame = imFeat.features.vl_frame(:,neighbourhoodMask == 1);
                if(~isempty(relevant_frame) && (params.completeNeighbourhood == true))
                  relative_orientations = relevant_frame(4,:) - imFeat.features.vl_frame(4,k);
                  %Make sure that the relative orientations are from 0 to
                  %2pi
                  relative_orientations(relative_orientations < 0) = relative_orientations(relative_orientations < 0) + (2*pi);
                  relative_orientations(relative_orientations > (2*pi)) = relative_orientations(relative_orientations > (2*pi)) - (2*pi);
%                  relative_distance2 = (relevant_frame(1,:) - features.data(1,k)).^2 + (relevant_frame(2,:) - features.data(2,k)).^2;
%                  relative_weight = exp(-relative_distance2/ (scale^2));
%                  relative_weight = relative_weight/norm(relative_weight);
                   relOrientationHist(k,:) = hist(relative_orientations, (pi/8:pi/4:2*pi))/length(relative_orientations);
                end

                if(~isempty(relevant_frame) && ((params.quarterNeighbourhood == true)||(params.halfNeighbourhood == true)))
                %Now that we have all the interesting keypoints, we would
                %transform the locations and filter the four quadrants and
                %then pool relative orientations in each quadrant
                 relative_locations = [relevant_frame(1,:) - x; relevant_frame(2,:) - y]; %Bring the keypoint to origin
                 rot = imFeat.features.vl_frame(4,k);
                 relative_locations = [cos(rot) sin(rot); -sin(rot) cos(rot)] * relative_locations;
                 quadMask = zeros(4,size(relative_locations,2));
                 quadMask(1,:) = quadMask(1,:) | ((relative_locations(1,:) >= 0 ) & (relative_locations(2,:) >=0 ));
                 quadMask(2,:) = quadMask(2,:) | ( (relative_locations(1,:) <= 0 ) & (relative_locations(2,:) >= 0) );
                 quadMask(3,:) = quadMask(3,:) | ( (relative_locations(1,:) <= 0 ) & (relative_locations(2,:) <= 0) );
                 quadMask(4,:) = quadMask(4,:) | ( (relative_locations(1,:) >= 0 ) & (relative_locations(2,:) <= 0) );
                 
                 for quad = 1:4
                     if(sum(quadMask(quad,:)>0))
                         quartNeighMask(quad,k,neighMask(k,:)==1) = quadMask(quad,:); 
                         relative_orientations = relevant_frame(4,quadMask(quad,:)==1) - rot;
                         relative_orientations(relative_orientations < 0) = relative_orientations(relative_orientations < 0) + (2*pi);
                         relative_orientations(relative_orientations > (2*pi)) = relative_orientations(relative_orientations > (2*pi)) - (2*pi);
                         quartRelOrientationHist(quad,k,:) = hist(relative_orientations, (pi/8:pi/4:2*pi))/length(relative_orientations);
                     end
                 end

                 halfMask = zeros(4,size(relative_locations,2));
                 halfMask(1,:) = quadMask(1,:)|quadMask(2,:);
                 halfMask(2,:) = quadMask(2,:)|quadMask(3,:);
                 halfMask(3,:) = quadMask(3,:)|quadMask(4,:);
                 halfMask(4,:) = quadMask(4,:)|quadMask(1,:);
                 for half = 1:4
                     if(sum(halfMask(half,:)>0))
                         halfNeighMask(half,k,neighMask(k,:)==1) = halfMask(half,:); 
                         relative_orientations = relevant_frame(4,halfMask(half,:)==1) - rot;
                         relative_orientations(relative_orientations < 0) = relative_orientations(relative_orientations < 0) + (2*pi);
                         relative_orientations(relative_orientations > (2*pi)) = relative_orientations(relative_orientations > (2*pi)) - (2*pi);
                         halfRelOrientationHist(half,k,:) = hist(relative_orientations, (pi/8:pi/4:2*pi))/length(relative_orientations);
                     end
                 end
                end
            end

            %Compute histograms now
            for dic = 1:length(params.keypointHistogramSize)
               coarseDic = params.keypointHistogramSize(dic);

               features.data = zeros(length(features.x), 8+coarseDic);
               features.data(:,1:8) = relOrientationHist;
                
               if(coarseDic ~= 0)
                   inFName = fullfile(dataBaseDir, sprintf('%s_soft_texton_ind_%d%s', baseFName, coarseDic, params.keypointFeatureSuffix));
                   load(inFName,'texton_ind');
               end

               outFName = fullfile(dataBaseDir, sprintf('%s_%dN_%dCD_ro_context.mat', baseFName, sigmaFactor, coarseDic));
               if(~((size(dir(outFName),1)~=0 && canSkip) || ~(params.completeNeighbourhood)) )
                   if(coarseDic == 0)
                    features.data = relOrientationHist;
                    save(outFName, 'features');
                   else
                    coarseHist = zeros(length(features.x),coarseDic);
                    for k = 1:length(features.x)
                        if(sum(neighMask(k,:)>0))
                            indices = texton_ind.data(neighMask(k,:)==1,:);
                            weights = texton_ind.weights(neighMask(k,:)==1,:);
                            indices = indices(:); weights = weights(:);
                            for n = 1:length(indices)
                                coarseHist(k, indices(n))= coarseHist(k, indices(n)) + weights(n);
                            end
                            coarseHist(k,:) = coarseHist(k,:)/(length(indices)/3); 
                        end
                    end
                    features.data(:,9:end) = coarseHist;
                    save(outFName, 'features');
                   end
               end

              quartOutFName = cell(4,1);
              halfOutFName = cell(4,1);
              for i = 1:4
                     quartOutFName{i,1} = fullfile(dataBaseDir, sprintf('%s_%dQ_%dN_%dCD_ro_context.mat', baseFName, i, sigmaFactor, coarseDic));
              end
              for i = 1:4
                     halfOutFName{i,1} = fullfile(dataBaseDir, sprintf('%s_%dH_%dN_%dCD_ro_context.mat', baseFName, i, sigmaFactor, coarseDic));
              end

              if(~ ((size(dir(quartOutFName{4,1}),1)~=0 && canSkip) || ~(params.quarterNeighbourhood)) )
                  for quad = 1:4
                    features.data(:,1:8) = reshape(quartRelOrientationHist(quad,:,:),size(features.data));
                    coarseHist = zeros(length(features.x),coarseDic);
                    if(coarseDic == 0)
                        features.data(:,1:8) = reshape(quartRelOrientationHist(quad,:,:),size(features.data));
                        save(quartOutFName{quad,1}, 'features');
                    else
                        for k = 1:length(features.x)
                              if(sum(quartNeighMask(quad, k,:))>0)
                                  indices = texton_ind.data(quartNeighMask(quad, k,:)==1,:);
                                  weights = texton_ind.weights(quartNeighMask(quad, k,:)==1,:);
                                  indices = indices(:); weights = weights(:);
                                  for n = 1:length(indices)
                                   coarseHist(k, indices(n))= coarseHist(k, indices(n)) + weights(n);
                                  end
                                  coarseHist(k,:) = coarseHist(k,:)/(length(indices)/3); 
                              end
                         end
                         features.data(:,9:end) = coarseHist;
                         save(quartOutFName{quad,1}, 'features');
                    end
                  end
              end

              if(~ ((size(dir(halfOutFName{4,1}),1)~=0 && canSkip) || ~(params.halfNeighbourhood)) )
                  for half = 1:4
                    features.data(:,1:8) = reshape(halfRelOrientationHist(half,:,:),size(features.data));
                    coarseHist = zeros(length(features.x),coarseDic);
                    if(coarseDic == 0)
                        features.data(:,1:8) = reshape(halfRelOrientationHist(half,:,:),size(features.data));
                        save(halfOutFName{half,1}, 'features');
                    else
                        for k = 1:length(features.x)
                            if(sum(halfNeighMask(half, k,:))>0)
                              indices = texton_ind.data(halfNeighMask(half, k,:)==1,:);
                              weights = texton_ind.weights(halfNeighMask(half, k,:)==1,:);
                              indices = indices(:); weights = weights(:);
                              for n = 1:length(indices)
                               coarseHist(k, indices(n))= coarseHist(k, indices(n)) + weights(n);
                              end
                              coarseHist(k,:) = coarseHist(k,:)/(length(indices)/3); 
                            end
                         end
                         features.data(:,9:end) = coarseHist;
                         save(halfOutFName{half,1}, 'features');
                    end
                  end
              end
            end  %Done for all coarse dictionary sizes
        
        end % For different sized neighbourhoods
    end % For All Images
end
