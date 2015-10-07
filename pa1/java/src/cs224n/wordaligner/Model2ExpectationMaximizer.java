package cs224n.wordaligner;

import cs224n.util.CounterMap;
import cs224n.util.Counters;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Arrays;
import java.util.List;

/**
 * Created by jcnnghm on 10/3/15.
 */
public class Model2ExpectationMaximizer extends Model1ExpectationMaximizer {
  private CounterMap<List<Integer>, Integer> q;
  private CounterMap<List<Integer>, Integer> qCount;

  public Model2ExpectationMaximizer(List<SentencePair> trainingPairs, CounterMap<String,String> initialT) {
    super(trainingPairs, initialT);
    q = setupQ(trainingPairs);
  }

  public static List<Integer> getQKey(SentencePair pair, int sourceIndex) {
    int l = pair.getSourceWords().size();
    int m = pair.getTargetWords().size();
    return Arrays.asList(sourceIndex, l, m);
  }

  protected void setupPosteriorCounts() {
    super.setupPosteriorCounts();
    qCount = new CounterMap<List<Integer>, Integer>();
  }

  @Override
  protected double calculateAlignmentProb(String sourceWord, String targetWord, int sourceIndex, int targetIndex, SentencePair pair) {
    return q.getCount(getQKey(pair, sourceIndex), targetIndex) * t.getCount(targetWord, sourceWord);
  }

  protected void updatePosteriorCounts(String targetWord, String sourceWord, int targetIndex, int sourceIndex, SentencePair pair,  double posterior) {
    super.updatePosteriorCounts(targetWord, sourceWord, targetIndex, sourceIndex, pair, posterior);
    qCount.incrementCount(getQKey(pair, sourceIndex), targetIndex, posterior);
  }

  protected void updateProbabilities() {
    super.updateProbabilities();
    CounterMap<List<Integer>, Integer> newQ = Counters.conditionalNormalize(qCount);
    q = newQ;
  }

  public CounterMap<List<Integer>, Integer> getQ() {
    return q;
  }

  private CounterMap<List<Integer>, Integer> setupQ(List<SentencePair> trainingPairs) {
    CounterMap<List<Integer>, Integer> q = new CounterMap<List<Integer>, Integer>();
    for (SentencePair pair : trainingPairs) {
      int sourceIndex = 0;
      for (String sourceWord : pair.getSourceWords()) {
        int targetIndex = 0;
        for (String targetWord : BaseAligner.targetWordsWithNull(pair)) {
          q.setCount(getQKey(pair, sourceIndex), targetIndex, 1.0);
          targetIndex++;
        }
        sourceIndex++;
      }
    }
    return Counters.conditionalNormalize(q);
  }
}
