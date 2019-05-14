function [] = visualizeNeighbourhood(image, ip_frame, num_keypoints, neighbourhood_frame)
    %clf
    hold on;
    if(exist('image','var'))
     imshow(image);
    end  
    
    if(exist('neighbourhood_frame','var'))
            plot(neighbourhood_frame.x(neighbourhood_frame.neighbourhood(3,:)>0),neighbourhood_frame.y(neighbourhood_frame.neighbourhood(3,:)>0), 'mo');
            plot(neighbourhood_frame.x(neighbourhood_frame.neighbourhood(3,:)>0),neighbourhood_frame.y(neighbourhood_frame.neighbourhood(3,:)>0), 'm*');
            plot(neighbourhood_frame.x(neighbourhood_frame.neighbourhood(2,:)>0),neighbourhood_frame.y(neighbourhood_frame.neighbourhood(2,:)>0), 'co');
            plot(neighbourhood_frame.x(neighbourhood_frame.neighbourhood(2,:)>0),neighbourhood_frame.y(neighbourhood_frame.neighbourhood(2,:)>0), 'c*');
            plot(neighbourhood_frame.x(neighbourhood_frame.neighbourhood(1,:)>0),neighbourhood_frame.y(neighbourhood_frame.neighbourhood(1,:)>0), 'ro');
            plot(neighbourhood_frame.x(neighbourhood_frame.neighbourhood(1,:)>0),neighbourhood_frame.y(neighbourhood_frame.neighbourhood(1,:)>0), 'r*');
    end
    
    if(exist('ip_frame','var'))
          num = min(num_keypoints, size(ip_frame,2));
          vl_plotframe(ip_frame(:,1:num));

    end
end
