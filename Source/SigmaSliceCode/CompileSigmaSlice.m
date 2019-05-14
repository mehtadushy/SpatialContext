function [sigmaSlices, visData] = CompileSigmaSlice( imageFileList, dataBaseDir, textonSuffix, interest_points, params, canSkip)
    if(~exist('params','var'))
        params.maxImageSize = 1000
        params.gridSpacing = 8
        params.patchSize = 16
        params.dictionarySize = 200
        params.numTextonImages = 50
        params.pyramidLevels = 3
        params.oldSift = false;
        params.sigmaFactors = [1 2 5]; 
        params.sigmaKeypoints = 15;
    end

    %This defines both the multiplication factors for sigma to define the
    %size of the neighbourhood, and the number of entries define the number
    %of containers across which the histograms are pooled
    if(~isfield(params,'sigmaFactors'))
        params.sigmaFactors = [1 2 5];
    end
    
    %This gives the maximum number of keypoints whose neigbourhood to add
    %to the pooling regions. If fewer interest points were detected in an
    %image, then you can't really do much
    if(~isfield(params,'sigmaKeypoints'))
        params.sigmaKeypoints = 15;
    end
    
    %This assumes that the features are already densely computed for the
    %images, a vocabulary built, and features assigned to the correcponding
    %textons
    %This also assumes that interest points have been computed, and the top 100
    %interest points (locations and sigma) are available for each image

    outFName = fullfile(dataBaseDir, sprintf('sigma_slice_%d_dictionary_%d_%s_keypoints_%d_%d_%d_scale.mat',params.dictionarySize, params.sigmaKeypoints, params.keypointDetector, params.sigmaFactors(1),params.sigmaFactors(2),params.sigmaFactors(3)));
    if(size(dir(outFName),1)~=0 && canSkip)
        load(outFName, 'sigmaSlices','visData');
        fprintf('File %s exits. Skipping recomputation of Sigma Slices for the current settings\n', outFName);
    else
         fprintf('Computing Sigma Slices for %d_dictionary_%d_%s_keypoints_%d_%d_%d_scale\n', params.dictionarySize, params.sigmaKeypoints, params.keypointDetector, params.sigmaFactors(1),params.sigmaFactors(2),params.sigmaFactors(3));

    sigmaSlices = zeros(length(imageFileList), params.dictionarySize*3);
    visData = cell(length(imageFileList), 1);
    
    for f = 1:length(imageFileList)
        
        if(mod(f,100)==0)
          %sp_progress_bar(pfig,4,4,f,length(imageFileList),'Compiling Sigma Slices:');
          fprintf('f = %d\n', f);
        end
        imageFName = imageFileList{f};
        [dirN base] = fileparts(imageFName);
        baseFName = fullfile(dirN, base);
        %Load texton indices
        in_fname = fullfile(dataBaseDir, sprintf('%s%s', baseFName, textonSuffix));
        load(in_fname, 'texton_ind');
        
        dataFrame.x = texton_ind.x;
        dataFrame.y = texton_ind.y;
        dataFrame.neighbourhood = zeros(3,length(texton_ind.x));

        %% Compute the mask that defines the neighbourhood for each scale multiple
        %  for all keypoints
        num_key = min(size(interest_points{f},2), params.sigmaKeypoints);
        for s = 1:3
            scaleFactor=params.sigmaFactors(s);
            neighbourhoodMask = zeros(size(texton_ind.data));
            
            for i = 1:num_key
                % find the coordinates of the current neighbourhood
                x = interest_points{f}(1,i);
                y = interest_points{f}(2,i);
                scale = 0.5 * scaleFactor * interest_points{f}(3,i);
                x_lo = x - scale;
                x_hi = x+scale;
                y_lo = y-scale;
                y_hi = y+scale;

                neighbourhoodMask = neighbourhoodMask | ( (texton_ind.x > x_lo) & (texton_ind.x <= x_hi) & ...
                                                (texton_ind.y > y_lo) & (texton_ind.y <= y_hi));
            end
            dataFrame.neighbourhood(s,:) = neighbourhoodMask;
            texton_region = texton_ind.data(neighbourhoodMask == 1);
                
            sigmaSlices(f, (s-1)*params.dictionarySize + 1 : s*params.dictionarySize) = hist(texton_region, 1:params.dictionarySize)./length(texton_region);
        end
        visData{f} = dataFrame;
    end
    save(outFName, 'sigmaSlices','visData');
    end
end
        


