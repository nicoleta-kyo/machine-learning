function [net, performance] = runNN(input, target, trainFcn, hiddenLayerSize) 

            net = fitnet(hiddenLayerSize,trainFcn);
            net.trainParam.epochs = 1000;
            net.trainParam.showWindow = true;
            net.trainParam.max_fail = 0;
            %set division function
            net.divideFcn = 'divideind';
            %regulate mu
            net.trainParam.mu = 0.5;
            net.trainParam.mu_dec = 0.4;
            net.trainParam.mu_inc = 6;
            % Setup Division of Data for Training and Testing
            [trainInd, testInd] = divideind(1000, 1:800, 801:1000);
            net.divideParam.trainInd = trainInd;
            %net.divideParam.valInd = [601:800];
            net.divideParam.testInd = testInd;
            %pre-processing
            net.input.processFcns = {'mapminmax', 'removeconstantrows'};
            net.output.processFcns = {'mapminmax', 'removeconstantrows'};
            
            % Train the Network
            [net,tr] = train(net,input, target);
            % Test the network on test set
              tInd = tr.testInd;
              tstOutputs = net(input(:, tInd));
              performance = perform(net, target(tInd), tstOutputs)
            % Test the network on all three sets
%             output = net(input);
%             performance = perform(net, target, output)

end
