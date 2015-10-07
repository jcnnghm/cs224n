package cs224n.wordaligner;

import cs224n.util.CounterMap;
import cs224n.util.Counters;

import java.util.List;

/**
 * Created by jcnnghm on 10/3/15.
 */
public class Model1ExpectationMaximizer {
  protected CounterMap<String,String> t;
  protected List<SentencePair> trainingPairs;
  private CounterMap<String, String> counts;
  protected double maxLastIterationProbChange;

  public Model1ExpectationMaximizer(List<SentencePair> trainingPairs) {
    this(trainingPairs, setupT(trainingPairs));
  }

  public Model1ExpectationMaximizer(List<SentencePair> trainingPairs, CounterMap<String,String> initialT) {
    this.trainingPairs = trainingPairs;
    this.t = initialT;
  }

  public void runUntilConvergence(double maxChangeForConvergence) {
    int iteration = 0;
    do {
      System.out.print("Starting iteration ");
      System.out.println(iteration);

      setupPosteriorCounts();

      int sourceIndex;
      int targetIndex;

      for (SentencePair pair : trainingPairs) {
        sourceIndex = 0;
        for (String sourceWord : pair.getSourceWords()) {
          double divisor = 0;
          targetIndex = 0;
          for (String targetWord : BaseAligner.targetWordsWithNull(pair)) {
            divisor += calculateAlignmentProb(sourceWord, targetWord, sourceIndex, targetIndex, pair);
            targetIndex++;
          }

          targetIndex = 0;
          for (String targetWord : BaseAligner.targetWordsWithNull(pair)) {
            double posterior = calculateAlignmentProb(sourceWord, targetWord, sourceIndex, targetIndex, pair) / divisor;
            updatePosteriorCounts(targetWord, sourceWord, targetIndex, sourceIndex, pair, posterior);
            targetIndex++;
          }
          sourceIndex++;
        }
      }

      updateProbabilities();

      System.out.print("Change: ");
      System.out.println(maxLastIterationProbChange);

      iteration++;
    } while(maxLastIterationProbChange > maxChangeForConvergence && iteration < 10);
  }

  public CounterMap<String, String> getT() {
    return t;
  }

  protected double calculateAlignmentProb(String sourceWord, String targetWord, int sourceIndex, int targetIndex, SentencePair pair) {
    return t.getCount(targetWord, sourceWord);
  }

  protected void setupPosteriorCounts() {
    counts = new CounterMap<String, String>();
  }

  protected void updatePosteriorCounts(String targetWord, String sourceWord, int targetIndex, int sourceIndex, SentencePair pair, double posterior) {
    counts.incrementCount(targetWord, sourceWord, posterior);
  }

  protected void updateProbabilities() {
    CounterMap<String, String> newT;
    newT = Counters.conditionalNormalize(counts);
    maxLastIterationProbChange = maxChange(t, newT);
    t = newT;
  }

  protected <D, E> double maxChange(CounterMap<D, E> originalValue, CounterMap<D, E> newValue) {
    double maxChange = 0;

    for (D key : newValue.keySet()) {
      for (E value : newValue.getCounter(key).keySet()) {
        maxChange = Math.max(maxChange, Math.abs(originalValue.getCount(key, value) - newValue.getCount(key, value)));
      }
    }

    return maxChange;
  }

  private static CounterMap<String, String> setupT(List<SentencePair> trainingPairs) {
    CounterMap<String, String> t = new CounterMap<String, String>();
    for (SentencePair pair : trainingPairs) {
      for (String sourceWord : pair.getSourceWords()) {
        for (String targetWord : BaseAligner.targetWordsWithNull(pair)) {
          t.setCount(targetWord, sourceWord, 1.0);
        }
      }
    }
    return Counters.conditionalNormalize(t);
  }
}