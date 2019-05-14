function [] = CreateKeypointSIFTContextFeatures(imageFileList, dataBaseDir, featureSuffix, params, canSkip)
    %This function puts the keypoint context features in a different file.
    %That's all

    
   fprintf('Compiling Keypoint SIFT Relative Context Features for \n');
%%
    
    for f = 1:length(imageFileList)
        
        imageFName = imageFileList{f};
        [dirN base] = fileparts(imageFName);
        baseFName = [dirN filesep base];
        %Load context file for each image
        in_fname = fullfile(dataBaseDir, sprintf('%s%s', baseFName, featureSuffix));
                          
        if(mod(f,100)==0)
           %sp_progress_bar(pfig,4,4,f,length(imageFileList),'Compiling Sigma Slices:');
           fprintf('f = %d\n', f);
        end
        
        outFName = fullfile(dataBaseDir, sprintf('%s_only_%s', baseFName, featureSuffix));
        
        if(size(dir(outFName),1)~=0 && canSkip)
            continue
        end
        
        load(in_fname, 'features');
        keypt_context = features;
        clear features;
        features.x = keypt_context.x;
        features.y = keypt_context.y;
        features.wid = keypt_context.wid;
        features.hgt = keypt_context.hgt;
        features.data = [keypt_context.spatialHist keypt_context.orientationHist]/2;
                 
        save(outFName, 'features');
        
    end
end

        


