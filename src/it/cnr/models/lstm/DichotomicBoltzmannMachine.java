package it.cnr.models.lstm;

import java.awt.GridLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import it.cnr.models.MultiLayerNetworkWhiteBox;

public class DichotomicBoltzmannMachine {
	private static final Logger log = LoggerFactory.getLogger(DichotomicBoltzmannMachine.class);

	public static void main(String[] args) throws Exception {
		test();
	}

	public static void test() throws Exception {
		String input = "./training_test_set_examples/linear_data_train_4BM.csv";
		List<String> allStr = Files.readAllLines(new File(input).toPath());
		double featureMatrix[][] = new double[allStr.size()][];
		double minPercAnomaly = 5; //0-100
		boolean binaryData = true; 

		int i = 0;
		for (String s : allStr) {
			String[] split = s.split(",");
			double[] vector = new double[split.length];
			for (int j = 0; j < vector.length; j++) {
				vector[j] = Double.parseDouble(split[j]);
			}
			featureMatrix[i] = vector;
			i++;
		}
		double anomalies[] = detectAnomaly(featureMatrix, minPercAnomaly,binaryData);
		System.out.println("Anomalies: " + Arrays.toString(anomalies));
	}

	// featureMatrix: one row for each observation, one column for each feature
	public static double[] detectAnomaly(double[][] featureMatrix, double minAnomalyPercentage, boolean binaryIO) throws Exception {

		int minibatchSize = featureMatrix.length;
		int rngSeed = 12345;
		int nEpochs = 100; // Total number of training epochs
		int reconstructionNumSamples = 16; // Reconstruction probabilities are estimated using Monte-Carlo techniques;
											// see An & Cho for details
		int nFeatures = featureMatrix[0].length;
		
		INDArray arr = Nd4j.create(featureMatrix);
		org.nd4j.linalg.dataset.DataSet d = new org.nd4j.linalg.dataset.DataSet(arr, arr);
		DataSetIterator trainIter = new ViewIterator(d, minibatchSize);

		// Neural net configuration
		Nd4j.getRandom().setSeed(rngSeed);
		MultiLayerConfiguration conf = null;
		if (binaryIO) {
		conf = new NeuralNetConfiguration.Builder().seed(rngSeed).updater(new Adam(1e-3))
				.weightInit(WeightInit.XAVIER).l2(1e-4).list()
				.layer(new VariationalAutoencoder.Builder().activation(Activation.LEAKYRELU).encoderLayerSizes(2)// .encoderLayerSizes(2,2)
																													// //2
																													// encoder
																													// layers,
																													// each
																													// of
																													// size
																													// 256
						.decoderLayerSizes(1) // 2 decoder layers, each of size 256
						.pzxActivationFunction(Activation.IDENTITY) // p(z|data) activation function
						// Bernoulli reconstruction distribution + sigmoid activation - for modelling
						// binary data (or data in range 0 to 1)
						.reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
						.nIn(nFeatures)
						.nOut(2).build())
				.build();
		}else {
			conf = new NeuralNetConfiguration.Builder().seed(rngSeed).updater(new Adam(1e-3))
					.weightInit(WeightInit.XAVIER).l2(1e-4).list()
					.layer(new VariationalAutoencoder.Builder().activation(Activation.LEAKYRELU).encoderLayerSizes(2)// .encoderLayerSizes(2,2)
																														// //2
																														// encoder
																														// layers,
																														// each
																														// of
																														// size
																														// 256
							.decoderLayerSizes(1) // 2 decoder layers, each of size 256
							.pzxActivationFunction(Activation.IDENTITY) // p(z|data) activation function
							.reconstructionDistribution(new GaussianReconstructionDistribution(Activation.TANH))
							.nIn(nFeatures)
							.nOut(2).build())
					.build();
		}
			
		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		// start learning
		log.trace(net.summary());

		System.out.println("Starting training...");
		// net.setListeners(new ScoreIterationListener(1), new
		// EvaluativeListener(trainIter, 100, InvocationType.EPOCH_END)); //Print the
		// score (loss function value) every 20 iterations
		net.setListeners(new ScoreIterationListener(nEpochs));
		// Fit the data (unsupervised training)
		for (int i = 0; i < nEpochs; i++) {
			net.pretrain(trainIter); // Note use of .pretrain(DataSetIterator) not fit(DataSetIterator) for
										// unsupervised training
			// System.out.println("Finished epoch " + (i+1) + " of " + nEpochs);
		}

		System.out.println("Training OK");

		// Get the variational autoencoder layer:
		org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net
				.getLayer(0);

		System.out.println("Testing");
		trainIter.reset();
		
		
		List<Double> scores = new ArrayList<>();
		List<Integer> featureOriginalIndices = new ArrayList<>();

		DataSet ds = trainIter.next();
		INDArray features = ds.getFeatures();
		int nRows = features.rows();
		INDArray reconstructionErrorEachExample = vae.reconstructionLogProbability(features, reconstructionNumSamples);
		
		for (int j = 0; j < nRows; j++) {
			INDArray example = features.getRow(j, true);
			// int label = (int)labels.getDouble(j);
			double score = reconstructionErrorEachExample.getDouble(j);
			System.out.println("\nf" + j + "->" + example);
			System.out.println("score" + j + "->" + score);
			INDArray realout = net.output(example);
			System.out.println("out" + j + "->" + realout);
			LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();
			// INDArray pzxMeanWorst = vae.preOutput(false, mgr);
			// vae.setInput(example, mgr);
			// INDArray pzxMean = vae.preOutput(false, mgr);
			INDArray reconstruction = vae.generateRandomGivenZ(realout, mgr);
			System.out.println("reconstructed" + j + "->" + reconstruction);
			int k = 0;
			int kbest = 0;
			for (Double sc : scores) {
				if (score < sc) {
					kbest = k;
					break;
				}
				k++;
			}
			
			scores.add(kbest, score);
			featureOriginalIndices.add(kbest, j);
		}
		
		double bestScore = scores.get(scores.size()-1); 
		double outliers[] = new double[featureMatrix.length];

		for (int g = 0; g < featureMatrix.length; g++) {

			Integer fi = featureOriginalIndices.get(g);
			double percAnomaly = Math.abs(scores.get(g)-bestScore)*100d/Math.abs(bestScore);
			if (percAnomaly>minAnomalyPercentage)
				outliers[fi.intValue()] = 1;
			else
				outliers[fi.intValue()] = 0;
		}

		return outliers;
	}

	public static void main2(String[] args) throws Exception {
		int minibatchSize = 128;
		int rngSeed = 12345;
		int nEpochs = 5; // Total number of training epochs
		int reconstructionNumSamples = 16; // Reconstruction probabilities are estimated using Monte-Carlo techniques;
											// see An & Cho for details

		int labelIndex = 4; // 5 values in each row of the animals.csv CSV: 4 input features followed by an
							// integer label (class) index. Labels are the 5th value (index 4) in each row
		int numClasses = 3; // 3 classes (types of animals) in the animals data set. Classes have integer
							// values 0, 1 or 2

		String input = "./training_test_set_examples/linear_data_train_4BM.csv";

		RecordReader rr = new CSVRecordReader();
		rr.initialize(new FileSplit(new File(input)));
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, minibatchSize, -1, -1);

		// Neural net configuration
		Nd4j.getRandom().setSeed(rngSeed);
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed).updater(new Adam(1e-3))
				.weightInit(WeightInit.XAVIER).l2(1e-4).list()
				.layer(new VariationalAutoencoder.Builder().activation(Activation.LEAKYRELU).encoderLayerSizes(2)// .encoderLayerSizes(2,2)
																													// //2
																													// encoder
																													// layers,
																													// each
																													// of
																													// size
																													// 256
						.decoderLayerSizes(1) // 2 decoder layers, each of size 256
						.pzxActivationFunction(Activation.IDENTITY) // p(z|data) activation function
						// Bernoulli reconstruction distribution + sigmoid activation - for modelling
						// binary data (or data in range 0 to 1)
						.reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID)).nIn(3)
						.nOut(2).build())
				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		// start learning
		log.trace(net.summary());

		System.out.println("Starting training...");
		net.setListeners(new ScoreIterationListener(1), new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END)); // Print
																															// the
																															// score
																															// (loss
																															// function
																															// value)
																															// every
																															// 20
																															// iterations

		// Fit the data (unsupervised training)
		for (int i = 0; i < nEpochs; i++) {
			net.pretrain(trainIter); // Note use of .pretrain(DataSetIterator) not fit(DataSetIterator) for
										// unsupervised training
			System.out.println("Finished epoch " + (i + 1) + " of " + nEpochs);
		}

		System.out.println("Training OK");

		// Get the variational autoencoder layer:
		org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net
				.getLayer(0);

		System.out.println("Testing");
		trainIter.reset();
		int idxf = 0;
		while (trainIter.hasNext()) {
			System.out.println("BATCH");
			DataSet ds = trainIter.next();
			INDArray features = ds.getFeatures();

			// INDArray labels = Nd4j.argMax(ds.getLabels(), 1); //Labels as integer indexes
			// (from one hot), shape [minibatchSize, 1]
			// System.out.println("Labels "+labels);
			int nRows = features.rows();
			INDArray reconstructionErrorEachExample = vae.reconstructionLogProbability(features,
					reconstructionNumSamples); // Shape: [minibatchSize, 1]
			for (int j = 0; j < nRows; j++) {
				INDArray example = features.getRow(j, true);
				// int label = (int)labels.getDouble(j);
				double score = reconstructionErrorEachExample.getDouble(j);
				System.out.println("\nf" + j + "->" + example);
				System.out.println("score" + j + "->" + score);
				INDArray realout = net.output(example);
				System.out.println("out" + j + "->" + realout);
				LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();
				// INDArray pzxMeanWorst = vae.preOutput(false, mgr);
				// vae.setInput(example, mgr);
				// INDArray pzxMean = vae.preOutput(false, mgr);
				INDArray reconstruction = vae.generateRandomGivenZ(realout, mgr);
				System.out.println("reconstructed" + j + "->" + reconstruction);
			}

			/*
			 * //Calculate the log probability for reconstructions as per An & Cho //Higher
			 * is better, lower is worse INDArray reconstructionErrorEachExample =
			 * vae.reconstructionLogProbability(features, reconstructionNumSamples);
			 * //Shape: [minibatchSize, 1] for( int j=0; j<nRows; j++){ INDArray example =
			 * features.getRow(j, true); int label = (int)labels.getDouble(j); double score
			 * = reconstructionErrorEachExample.getDouble(j);
			 * listsByDigit.get(label).add(new Pair<>(score, example)); }
			 */
			idxf++;
		}

		// Perform anomaly detection on the test set, by calculating the reconstruction
		// probability for each example
		// Then add pair (reconstruction probability, INDArray data) to lists and sort
		// by score
		// This allows us to get best N and worst N digits for each digit type

	}

	public static void main1(String[] args) throws IOException {
		int minibatchSize = 128;
		int rngSeed = 12345;
		int nEpochs = 5; // Total number of training epochs
		int reconstructionNumSamples = 16; // Reconstruction probabilities are estimated using Monte-Carlo techniques;
											// see An & Cho for details

		// MNIST data for training
		DataSetIterator trainIter = new MnistDataSetIterator(minibatchSize, true, rngSeed);
		int idx = 0;
		while (trainIter.hasNext()) {
			DataSet ds = trainIter.next();
			System.out.println("F" + idx + ": " + ds.getFeatures());
			idx++;
		}

		trainIter.reset();
		// Neural net configuration
		Nd4j.getRandom().setSeed(rngSeed);
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed).updater(new Adam(1e-3))
				.weightInit(WeightInit.XAVIER).l2(1e-4).list()
				.layer(new VariationalAutoencoder.Builder().activation(Activation.LEAKYRELU).encoderLayerSizes(256, 256) // 2
																															// encoder
																															// layers,
																															// each
																															// of
																															// size
																															// 256
						.decoderLayerSizes(256, 256) // 2 decoder layers, each of size 256
						.pzxActivationFunction(Activation.IDENTITY) // p(z|data) activation function
						// Bernoulli reconstruction distribution + sigmoid activation - for modelling
						// binary data (or data in range 0 to 1)
						.reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
						.nIn(28 * 28) // Input size: 28x28
						.nOut(32) // Size of the latent variable space: p(z|x) - 32 values
						.build())
				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		net.setListeners(new ScoreIterationListener(100));

		// Fit the data (unsupervised training)
		for (int i = 0; i < nEpochs; i++) {
			net.pretrain(trainIter); // Note use of .pretrain(DataSetIterator) not fit(DataSetIterator) for
										// unsupervised training
			System.out.println("Finished epoch " + (i + 1) + " of " + nEpochs);
		}

		// Perform anomaly detection on the test set, by calculating the reconstruction
		// probability for each example
		// Then add pair (reconstruction probability, INDArray data) to lists and sort
		// by score
		// This allows us to get best N and worst N digits for each digit type

		DataSetIterator testIter = new MnistDataSetIterator(minibatchSize, false, rngSeed);

		// Get the variational autoencoder layer:
		org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net
				.getLayer(0);

		Map<Integer, List<Pair<Double, INDArray>>> listsByDigit = new HashMap<>();
		for (int i = 0; i < 10; i++)
			listsByDigit.put(i, new ArrayList<>());

		// Iterate over the test data, calculating reconstruction probabilities
		while (testIter.hasNext()) {
			DataSet ds = testIter.next();
			INDArray features = ds.getFeatures();

			INDArray labels = Nd4j.argMax(ds.getLabels(), 1); // Labels as integer indexes (from one hot), shape
																// [minibatchSize, 1]
			int nRows = features.rows();

			// Calculate the log probability for reconstructions as per An & Cho
			// Higher is better, lower is worse
			INDArray reconstructionErrorEachExample = vae.reconstructionLogProbability(features,
					reconstructionNumSamples); // Shape: [minibatchSize, 1]
			// System.out.println("Output "+reconstructionErrorEachExample);
			for (int j = 0; j < nRows; j++) {
				INDArray example = features.getRow(j, true);
				int label = (int) labels.getDouble(j);
				double score = reconstructionErrorEachExample.getDouble(j);
				listsByDigit.get(label).add(new Pair<>(score, example));
			}
		}

		// Sort data by score, separately for each digit
		Comparator<Pair<Double, INDArray>> c = new Comparator<Pair<Double, INDArray>>() {
			@Override
			public int compare(Pair<Double, INDArray> o1, Pair<Double, INDArray> o2) {
				// Negative: return highest reconstruction probabilities first -> sorted from
				// best to worst
				return -Double.compare(o1.getFirst(), o2.getFirst());
			}
		};

		for (List<Pair<Double, INDArray>> list : listsByDigit.values()) {
			Collections.sort(list, c);
		}

		// Select the 5 best and 5 worst numbers (by reconstruction probability) for
		// each digit
		List<INDArray> best = new ArrayList<>(50);
		List<INDArray> worst = new ArrayList<>(50);

		List<INDArray> bestReconstruction = new ArrayList<>(50);
		List<INDArray> worstReconstruction = new ArrayList<>(50);
		for (int i = 0; i < 10; i++) {
			List<Pair<Double, INDArray>> list = listsByDigit.get(i);
			for (int j = 0; j < 5; j++) {
				INDArray b = list.get(j).getSecond();
				INDArray w = list.get(list.size() - j - 1).getSecond();

				LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();
				vae.setInput(b, mgr);
				INDArray pzxMeanBest = vae.preOutput(false, mgr);
				INDArray reconstructionBest = vae.generateAtMeanGivenZ(pzxMeanBest);

				vae.setInput(w, mgr);
				INDArray pzxMeanWorst = vae.preOutput(false, mgr);
				INDArray reconstructionWorst = vae.generateAtMeanGivenZ(pzxMeanWorst);

				best.add(b);
				bestReconstruction.add(reconstructionBest);
				worst.add(w);
				worstReconstruction.add(reconstructionWorst);
			}
		}

		// Visualize the best and worst digits
		MNISTVisualizer bestVisualizer = new MNISTVisualizer(2.0, best, "Best (Highest Rec. Prob)");
		bestVisualizer.visualize();

		MNISTVisualizer bestReconstructions = new MNISTVisualizer(2.0, bestReconstruction, "Best - Reconstructions");
		bestReconstructions.visualize();

		MNISTVisualizer worstVisualizer = new MNISTVisualizer(2.0, worst, "Worst (Lowest Rec. Prob)");
		worstVisualizer.visualize();

		MNISTVisualizer worstReconstructions = new MNISTVisualizer(2.0, worstReconstruction, "Worst - Reconstructions");
		worstReconstructions.visualize();
	}

	public static class MNISTVisualizer {
		private double imageScale;
		private List<INDArray> digits; // Digits (as row vectors), one per INDArray
		private String title;
		private int gridWidth;

		public MNISTVisualizer(double imageScale, List<INDArray> digits, String title) {
			this(imageScale, digits, title, 5);
		}

		public MNISTVisualizer(double imageScale, List<INDArray> digits, String title, int gridWidth) {
			this.imageScale = imageScale;
			this.digits = digits;
			this.title = title;
			this.gridWidth = gridWidth;
		}

		public void visualize() {
			JFrame frame = new JFrame();
			frame.setTitle(title);
			frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

			JPanel panel = new JPanel();
			panel.setLayout(new GridLayout(0, gridWidth));

			List<JLabel> list = getComponents();
			for (JLabel image : list) {
				panel.add(image);
			}

			frame.add(panel);
			frame.setVisible(true);
			frame.pack();
		}

		private List<JLabel> getComponents() {
			List<JLabel> images = new ArrayList<>();
			for (INDArray arr : digits) {
				BufferedImage bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
				for (int i = 0; i < 784; i++) {
					bi.getRaster().setSample(i % 28, i / 28, 0, (int) (255 * arr.getDouble(i)));
				}
				ImageIcon orig = new ImageIcon(bi);
				Image imageScaled = orig.getImage().getScaledInstance((int) (imageScale * 28), (int) (imageScale * 28),
						Image.SCALE_REPLICATE);
				ImageIcon scaled = new ImageIcon(imageScaled);
				images.add(new JLabel(scaled));
			}
			return images;
		}
	}
}
