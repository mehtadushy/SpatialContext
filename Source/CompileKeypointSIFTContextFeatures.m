function [] = CompileKeypointSIFTContextFeatures(imageFileList, dataBaseDir, featureSuffix, params, canSkip)
    %This function compiles SIFT and its neighbourhood context in various
    %ways to produce (hopefully) more descriptive features
    if(~exist('params','var'))
      
        params.weights = [ %SIFT   %Spatial Context  %Orientation Context
                              1          1               0;
                              1          0               1;
                              1          1               1;
                              1          0.8             1;
                              0.5         0              1;
                              1           0              0.8];
    end


    if(~isfield(params,'weights'))

        params.weights = [ %SIFT   %Spatial Context  %Orientation Context
                              1          1               0;
                              1          0               1;
                              1          1               1;
                              1          0.8             1;
                              0.5         0              1;
                              1           0              0.8];
    end
    

    
   fprintf('Compiling Keypoint SIFT Relative Context Features for \n');
%%
    
    for f = 1:length(imageFileList)
        
        imageFName = imageFileList{f};
        [dirN base] = fileparts(imageFName);
        baseFName = [dirN filesep base];
        %Load context for each image
        in_fname = fullfile(dataBaseDir, sprintf('%s%s', baseFName, featureSuffix));
        load(in_fname, 'features');
        keypt_context = features;
        clear features;
        features.x = keypt_context.x;
        features.y = keypt_context.y;
        features.wid = keypt_context.wid;
        features.hgt = keypt_context.hgt;
                          
        if(mod(f,100)==0)
           %sp_progress_bar(pfig,4,4,f,length(imageFileList),'Compiling Sigma Slices:');
           fprintf('f = %d\n', f);
        end
        
        for w = 1:size(params.weights,1)
            sift_weight = params.weights(w,1);
            spatial_weight = params.weights(w,2);
            orientation_weight = params.weights(w,3);
            if (sift_weight > 0)
                sift_string = sprintf('_siftWght_%1.1f',sift_weight);
            else
                sift_string = '';
            end
            if (spatial_weight > 0)
                spatial_string = sprintf('_spatialWht_%1.1f',spatial_weight);
            else
                spatial_string = '';
            end
            if (orientation_weight > 0)
                orientation_string = sprintf('_orientationWht_%1.1f',orientation_weight);
            else
                orientation_string = '';
            end            

            outFName = fullfile(dataBaseDir, sprintf('%s%s%s%s%s', baseFName, sift_string, spatial_string, orientation_string, featureSuffix));
        
            if(size(dir(outFName),1)~=0 && canSkip)
                continue
            end
            
            weight_sum = sift_weight+ spatial_weight + orientation_weight;
            features.data = [];
            if(sift_weight>0)
                features.data = [features.data keypt_context.data*(sift_weight/weight_sum)];
            end
            if(spatial_weight>0)
                features.data = [features.data keypt_context.spatialHist*(spatial_weight/weight_sum)];
            end
            if(orientation_weight>0)
                features.data = [features.data keypt_context.orientationHist*(orientation_weight/weight_sum)];
            end

            
            save(outFName, 'features');
        end
    end
end

        


