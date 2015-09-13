package com.semantive.deeplearning.examples;

import org.canova.api.io.WritableConverter;
import org.canova.api.io.converters.WritableConverterException;
import org.canova.api.io.data.IntWritable;
import org.canova.api.io.data.Text;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.records.reader.impl.ComposableRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Cifar {

    private static final int WIDTH = 32;
    private static final int HEIGHT = 32;

    private static final int BATCH_SIZE = 100;
    private static final int ITERATIONS = 10;

    private static final int SEED = 123;

    private static final List<String> LABELS = Arrays.asList("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck");

    private static final Logger log = LoggerFactory.getLogger(Cifar.class);

    public static void main(String[] args) throws Exception {

        int splitTrainNum = (int) (BATCH_SIZE * 0.8);
        int listenerFreq = ITERATIONS / 5;

        DataSet cifarDataSet;
        SplitTestAndTrain trainAndTest;
        DataSet trainInput;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();

        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        RecordReader recordReader = loadData();
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, new WritableConverter() {
            @Override
            public Writable convert(Writable writable) throws WritableConverterException {
                if (writable instanceof Text) {
                    String label = writable.toString().replaceAll("\u0000", "");
                    int index = LABELS.indexOf(label);
                    return new IntWritable(index);
                }
                return writable;
            }
        }, BATCH_SIZE, 1024, 10);

        MultiLayerNetwork model = new MultiLayerNetwork(getConfiguration());
        model.init();

        log.info("Train model");
        while (dataSetIterator.hasNext()) {
            model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
            cifarDataSet = dataSetIterator.next();
            trainAndTest = cifarDataSet.splitTestAndTrain(splitTrainNum, new Random(SEED));
            trainInput = trainAndTest.getTrain();
            testInput.add(trainAndTest.getTest().getFeatureMatrix());
            testLabels.add(trainAndTest.getTest().getLabels());
            model.fit(trainInput);
        }

        log.info("Evaluate model");
        Evaluation eval = new Evaluation(LABELS.size());
        for (int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }

        log.info(eval.stats());
    }

    public static MultiLayerConfiguration getConfiguration() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .batchSize(BATCH_SIZE)
                .iterations(ITERATIONS)
                .momentum(0.9)
                .regularization(true)
                .constrainGradientToUnitNorm(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5})
                        .nIn(1)
                        .nOut(20)
                        .stride(new int[]{1, 1})
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{5, 5})
                        .nIn(20)
                        .nOut(40)
                        .stride(new int[]{1, 1})
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nIn(40 * 5 * 5)
                        .nOut(1000)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .dropOut(0.5)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(1000)
                        .nOut(LABELS.size())
                        .dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(WIDTH, HEIGHT, 1))
                .inputPreProcessor(4, new CnnToFeedForwardPreProcessor())
                .backprop(true).pretrain(false)
                .build();

        return conf;
    }


    public static RecordReader loadData() throws Exception {
        RecordReader imageReader = new ImageRecordReader(32, 32, false);
        imageReader.initialize(new FileSplit(new File(System.getProperty("user.home"), "/deep-learning/data/cifar/img")));

        RecordReader labelsReader = new CSVRecordReader();
        labelsReader.initialize(new FileSplit(new File(System.getProperty("user.home"), "/deep-learning/data/cifar/labels.csv")));

        return new ComposableRecordReader(imageReader, labelsReader);
    }
}
