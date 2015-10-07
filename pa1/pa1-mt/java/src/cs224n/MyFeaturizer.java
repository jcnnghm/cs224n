package edu.stanford.nlp.mt.decoder.feat;

import java.util.List;

import edu.stanford.nlp.mt.util.FeatureValue;
import edu.stanford.nlp.mt.util.Featurizable;
import edu.stanford.nlp.mt.util.IString;
import edu.stanford.nlp.mt.decoder.feat.RuleFeaturizer;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.mt.util.TokenUtils;

/**
 * A rule featurizer.
 */
public class MyFeaturizer implements RuleFeaturizer<IString, String> {

  private static final String FEATURE_NAME = "SIZE_CHANGE";

  @Override
  public void initialize() {
    // Do any setup here.
  }

  @Override
  public List<FeatureValue<String>> ruleFeaturize(
      Featurizable<IString, String> f) {

    List<FeatureValue<String>> features = Generics.newLinkedList();
    features.add(new FeatureValue<String>("bias", 1.0));

    int targetSize = f.targetPhrase.size();
    int sourceSize = f.sourcePhrase.size();

    features.add(new FeatureValue<String>(String.format("%s:%d", FEATURE_NAME, targetSize - sourceSize), 1.0));

    int punctuation = 0;
    for (IString token : f.sourcePhrase) {
      if (TokenUtils.isPunctuation(token.toString())) {
        punctuation++;
      }
    }
    for (IString token : f.targetPhrase) {
      if (TokenUtils.isPunctuation(token.toString())) {
        punctuation--;
      }
    }
    features.add(new FeatureValue<String>(String.format("PUNCUATION_BALANCE:%d", punctuation), 1.0));



    return features;
  }

  @Override
  public boolean isolationScoreOnly() {
    return false;
  }
}
