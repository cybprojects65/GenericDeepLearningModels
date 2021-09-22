/* *****************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package it.cnr.speech.acousticmodelling.experiments.lstm;

import java.io.File;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.guava.io.Files;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


@SuppressWarnings("ResultOfMethodCallIgnored")
public class TrainLSTM {
    private static final Logger log = LoggerFactory.getLogger(TrainLSTM.class);

    //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
    private static File baseDir = new File("D:\\WorkFolder\\Experiments\\LSTMJava\\LongShortTermMemory\\datafiles");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");

    public static void main(String[] args) throws Exception {
        //downloadUCIData();
    	new File("./logs/application.log").delete();
        // ----- Load the training data -----
        //Note that we have 450 training files for features: train/features/0.csv through train/features/449.csv
        int nfilesTraining = featuresDirTrain.listFiles().length-1;
        int nfilesTest = featuresDirTest.listFiles().length-1;
        int nhidden = 200;
        int nInput = 39;
        int numLabelClasses = 44;
        int miniBatchSize = 1000;
        int nEpochs = 85;
        File savedNet = new File("net_"+nhidden+"hidden_"+nEpochs+"epo_"+miniBatchSize+"batch_"+numLabelClasses+"class.bin");
        File savedNormalizer = new File("net_"+nhidden+"hidden_"+nEpochs+"epo_"+miniBatchSize+"batch_"+numLabelClasses+"class_normaliser.bin");
        
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, nfilesTraining));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, nfilesTraining));
        
        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //printDataSet(trainData);
        //System.exit(0);
        
        //Normalize the training data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);              //Collect training data statistics
        trainData.reset();
        
        //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
        trainData.setPreProcessor(normalizer);
        // ----- Load the test data -----
        //Same process as for the training data.
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, nfilesTest));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, nfilesTest));

        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data


        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(new LSTM.Builder().activation(Activation.TANH).nIn(nInput).nOut(nhidden).build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(nhidden).nOut(numLabelClasses).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        

        log.info("Learning rate "+net.getLearningRate(1));
        
        ModelSerializer.writeModel(net, savedNormalizer, false, normalizer);
        
        
        log.info("Starting training...");
        net.setListeners(new ScoreIterationListener(20), new EvaluativeListener(testData, 1, InvocationType.EPOCH_END));   //Print the score (loss function value) every 20 iterations

        
        net.fit(trainData, nEpochs);

        log.info("Evaluating...");
        Evaluation eval = net.evaluate(testData);
        log.info(eval.stats());
       
        net.save(savedNet);
        Files.copy(new File("./logs/application.log"), new File("./logs/",savedNet.getName()+".log"));
        
        log.info("----- Example Complete -----");
    }


    
    public static void printDataSet(DataSetIterator trainData) {
		trainData.reset();
    	int i=0;
    	while (trainData.hasNext()) {
    		DataSet data = trainData.next();
    		//System.out.println("Train data "+i+" - Labels: "+data.getLabels());
    		System.out.println("Train data "+i+";\n"+
    				"N. Inputs:"+data.numInputs()+";\n"+
    				"N. Examples:"+data.numExamples()+";\n"+
    				"N. Outputs:"+data.numOutcomes()+";\n"
    				
    				);
    		
    		for (int k=0;k<data.numExamples();k++) {
    			DataSet ex  = data.get(k);
    			System.out.println("Example "+k+";\n"+
    					"N. Inputs:"+ex.numInputs()+";\n"+
    					"N. Examples:"+ex.numExamples()+";\n"+
    					"N. Outputs:"+ex.numOutcomes()+";\n"
        				);
    			
    			
    			
    			NDArray inputArray = (NDArray)ex.getFeatures();
    			//String array = inputArray.toString();
    			
    			System.out.println("Input sample dimension:\n"+
    					"N. Rows:"+inputArray.size(0)+";\nN. Columns:"+inputArray.size(1)+";\nArray:\n"+inputArray
    			);
    			System.out.println("Outcome:\n"+ex.outcome());
    			System.out.println("Outcome:\n"+ex.getLabels());

    		}
    		
    		
    		i++;
    	}
	}
    
   }
