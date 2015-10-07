package cs224n.wordaligner;  

import cs224n.util.*;

import java.util.Arrays;
import java.util.List;

/**
 * Simple word alignment baseline model that maps source positions to target 
 * positions along the diagonal of the alignment grid.
 * 
 * IMPORTANT: Make sure that you read the comments in the
 * cs224n.wordaligner.WordAligner interface.
 * 
 * @author Dan Klein
 * @author Spence Green
 */
public class IBMModel2Aligner extends BaseAligner {

  private static final long serialVersionUID = 1315751943476440515L;
  private CounterMap<String,String> t;
  private CounterMap<List<Integer>, Integer> q;

  @Override
  protected double calculateAlignmentProb(String sourceWord, String targetWord, int sourceIndex, int targetIndex, SentencePair pair) {
    return q.getCount(Model2ExpectationMaximizer.getQKey(pair, sourceIndex), targetIndex) * t.getCount(targetWord, sourceWord);
  }

  public void train(List<SentencePair> trainingPairs) {
    Model1ExpectationMaximizer model1 = new Model1ExpectationMaximizer(trainingPairs);
    model1.runUntilConvergence(0.03);
    t = model1.getT();

    Model2ExpectationMaximizer model2 = new Model2ExpectationMaximizer(trainingPairs, t);
    model2.runUntilConvergence(0.02);
    t = model2.getT();
    q = model2.getQ();
  }

  public static void main(String[] args) {
    List<SentencePair> pairs = Arrays.asList(
            new SentencePair(1, "none", Arrays.asList("blue", "house"), Arrays.asList("maison", "bleu")),
            new SentencePair(2, "none", Arrays.asList("blue", "house"), Arrays.asList("maison", "bleu")),
            new SentencePair(3, "none", Arrays.asList("house"), Arrays.asList("maison"))
    );
    System.out.println(pairs);

    IBMModel2Aligner aligner = new IBMModel2Aligner();
    aligner.train(pairs);

    System.out.println(aligner.t);
    System.out.println(aligner.q);
  }
}