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
public class IBMModel1Aligner extends BaseAligner {

  // ENGLISH -> target
  // FRENCH -> source

  private static final long serialVersionUID = 1315751943476440515L;
  
  private CounterMap<String,String> t;

  protected double calculateAlignmentProb(String sourceWord, String targetWord) {
    // q(a|i,l,m) is uniform, so it's unnecessary to include it here, since it
    // won't impact the argmax that we select.
    return t.getCount(targetWord, sourceWord);
  }

  public void train(List<SentencePair> trainingPairs) {
    Model1ExpectationMaximizer maximizer = new Model1ExpectationMaximizer(trainingPairs);
    maximizer.runUntilConvergence(0.03);
    t = maximizer.getT();
  }

  public static void main(String[] args) {
    List<SentencePair> pairs = Arrays.asList(
            new SentencePair(1, "none", Arrays.asList("blue", "house"), Arrays.asList("maison", "bleu")),
            new SentencePair(2, "none", Arrays.asList("blue", "house"), Arrays.asList("maison", "bleu")),
            new SentencePair(3, "none", Arrays.asList("house"), Arrays.asList("maison"))
    );
    System.out.println(pairs);

    IBMModel1Aligner aligner = new IBMModel1Aligner();
    aligner.train(pairs);

    System.out.println(aligner.t);
  }
}