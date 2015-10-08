package edu.stanford.nlp.mt.decoder.feat;

import java.util.List;
import java.util.ArrayList;
import java.util.Properties;
import java.util.TreeSet;

import edu.stanford.nlp.mt.util.FeatureValue;
import edu.stanford.nlp.mt.util.Featurizable;
import edu.stanford.nlp.mt.util.IString;
import edu.stanford.nlp.mt.decoder.feat.RuleFeaturizer;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.mt.util.TokenUtils;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.ling.*;

/**
 * A rule featurizer.
 */
public class MyFeaturizer implements RuleFeaturizer<IString, String> {

  private static final String FEATURE_NAME = "SIZE_CHANGE";
  private StanfordCoreNLP pipeline;
  private StanfordCoreNLP frenchPipeline;

  @Override
  public void initialize() {
    Properties props = new Properties();
    props.setProperty("annotators", "tokenize, ssplit, pos");
    pipeline = new StanfordCoreNLP(props);

    Properties frenchProps = new Properties();
    frenchProps.setProperty("annotators", "tokenize, ssplit, pos");
    frenchProps.setProperty("pos.model", "french.tagger");
    frenchPipeline = new StanfordCoreNLP(frenchProps);
  }

  @Override
  public List<FeatureValue<String>> ruleFeaturize(
      Featurizable<IString, String> f) {

    List<FeatureValue<String>> features = Generics.newLinkedList();

    List<String> posTags = getPosTags(pipeline, f.targetPhrase.toString());
    List<String> frenchPosTags = getPosTags(frenchPipeline, f.sourcePhrase.toString());

    features.add(new FeatureValue<String>("bias", 1.0));

    for (int i=-2; i<posTags.size(); i++) {
      if (i>=0) addFeature(features, String.format("TARGET_POS:%s", getPos(posTags, i)));
      if (i>=-1) addFeature(features, String.format("TARGET_POS_BIGRAM:%s:%s", getPos(posTags, i), getPos(posTags, i + 1)));
      addFeature(features, String.format("TARGET_POS_TRIGRAM:%s:%s:%s", getPos(posTags, i), getPos(posTags, i + 1), getPos(posTags, i + 2)));
    }

    TreeSet<String> englishTags = new TreeSet<String>(posTags);
    TreeSet<String> frenchTags = new TreeSet<String>(frenchPosTags);

    for (String englishTag : englishTags) {
      for (String frenchTag : frenchTags) {
        addFeature(features, String.format("POS_CO_OCCUR:%s:%s", englishTag, frenchTag));
      }
    }

    // The sets are sorted, so this order is consistent regardless of appearance order
    addFeature(features, String.format("POS_CHANGE:%s:%s", join(englishTags, "|"), join(frenchTags, "|")));

    return features;
  }

  private void addFeature(List<FeatureValue<String>> features, String featureName) {
    features.add(new FeatureValue<String>(featureName, 1.0));
  }

  private String getPos(List<String> posTags, int index) {
    if (index < 0) return "NULL";
    if (index >= posTags.size()) return "NULL";
    return posTags.get(index);
  }

  private List<String> getPosTags(StanfordCoreNLP pipeline, String phrase) {
    ArrayList<String> partsOfSpeech = new ArrayList<String>();
    Annotation document = new Annotation(phrase);
    pipeline.annotate(document);
    List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
    for(CoreMap sentence: sentences) {
      for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
        String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
        partsOfSpeech.add(pos);
      }
    }
    return partsOfSpeech;
  }

  private String join(TreeSet<String> strings, String joinString) {
    StringBuilder builder = new StringBuilder();
    int count = 0;
    for (String str : strings) {
      if (count!=0) builder.append(joinString);
      builder.append(str);
      count++;
    }
    return builder.toString();
  }

  @Override
  public boolean isolationScoreOnly() {
    return false;
  }
}
