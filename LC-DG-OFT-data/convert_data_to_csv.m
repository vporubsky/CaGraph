%% Script for looking at matrices, converting data to .csv

load mouseData_CNOPraz.mat
load caData_CNOPraz.mat

load caData_ThDay9.mat

% load mouseData_CNOQuetiapine5mgkg.mat
% load caData_CNOQuetiapine5mgkg.mat

% load mouseData_CNOSaline.mat
% load caData_CNOSaline.mat

% load mouseData_Praz.mat
% load caData_Praz.mat
% 
% load mouseData_Prop.mat
% load caData_Prop.mat

% load mouseData_Quetiapine5mgkg.mat
% load caData_Quetiapine5mgkg.mat

% load mouseData_Quetiapine10mgkg.mat
% load caData_Quetiapine10mgkg.mat

% load mouseData_Saline.mat
% load caData_Saline.mat


%% Test plot
% y = mouseData(2).nonSelectiveEventData(1).eventTrace(1,1:36000);
% x = linspace(0, 360,  36000);
% y2 = mouseData(1).caData.rawCaTrace(1,1:36000);
% 
% hold on 
% plot(x, y)
% plot(x, y2)
% hold off

%% Export csv files of calcium dynamics
% for i = 1:12
%     cellData = caData(i).data;
%     allCellData = caData(i).caTraceInfo.timeVector(1,1:36000);
%     for c = 1:length(caData(i).data)
%         allCellData = [allCellData; cellData(c).rawCaTrace];
%     end
%     s1 = erase(caData(i).caFileName , '_Day2.CNMF_final.mat' );
%     s2 =  '_D1_raw_calcium_traces.csv';
%     filename = strcat(s1,s2);
%     % export data to csv, where each row is a timecourse calcium trace, first row is time
%     csvwrite(filename, allCellData);    
% end

% %% Export csv files of context activity assignment
% 
% for i = 1:12
%     activityData = [];
%     for c  = 1:length(mouseData(i).caData)
%         activityData = [activityData, mouseData(i).caData(c).dayOneSelectivity];
%     end
% 
%     s1 = erase(mouseData(i).name , 'Day1CNMF_final.mat' );
%     s2 =  '_D1_neuron_context_active.csv';
%     filename = strcat(s1,s2);
%     % export data to csv, where each row is a timecourse calcium trace, first row is time
%     csvwrite(filename, activityData);    
% end

%% Export smoothed trace csv files all timepoints
for i = 1:2
    cellData = caData(i).data;
    allCellData = caData(i).caTraceInfo.timeVector;
    for c = 1:length(caData(i).data)
        allCellData = [allCellData; cellData(c).caTrace];
    end
    s1 = erase(caData(i).caFileName , '.CNMF_final.mat' );
    s2 =  '.csv';
    filename = strcat(s1,s2);
    % export data to csv, where each row is a timecourse calcium trace, first row is time
    csvwrite(filename, allCellData);    
end

