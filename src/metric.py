import numpy as np
from llama_index.core.base.embeddings.base import SimilarityMode
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.core.service_context import ServiceContext
from model import Embedding, LLM

from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency

class QuestionMetric:
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def calculate_jsd(self, counts):
        """
        Calculate the Jensen-Shannon Divergence between the given counts and a uniform distribution.

        Parameters:
        - counts: List or numpy array of counts.

        Returns:
        - jsd: Jensen-Shannon Divergence between counts and uniform distribution.
        """
        # Convert to numpy array
        counts = np.array(counts, dtype=np.float64)

        # Calculate the total counts and total number of nodes
        total_count = counts.sum()
        total_nodes = len(counts)

        # Calculate the uniform probability for each key
        uniform_count = total_count / total_nodes

        # Create a new probability distribution with the same number of keys
        probability_distribution = np.array([uniform_count] * total_nodes)

        # To avoid log(0), we add a small value (epsilon) to the counts and uniform distributions
        counts += self.epsilon
        probability_distribution += self.epsilon

        # Normalize to get probabilities
        prob = counts / counts.sum()
        uniform_prob = probability_distribution / probability_distribution.sum()

        # Calculate Jensen-Shannon Divergence
        jsd = jensenshannon(prob, uniform_prob, base=2)

        return jsd

    def calculate_chi_square(self, counts):
        """
        Calculate the Chi-Square Goodness of Fit test statistic and p-value.

        Parameters:
        - counts: List or numpy array of observed counts.

        Returns:
        - chi2: Chi-Square test statistic.
        - p: p-value of the test.
        """
        counts = np.array(counts, dtype=np.float64)
        total_count = counts.sum()
        total_nodes = len(counts)
        uniform_count = total_count / total_nodes
        expected_counts = np.array([uniform_count] * total_nodes, dtype=np.float64)
        chi2, p = chi2_contingency([counts, expected_counts])[:2]
        return chi2, p

    def calculate_tvd(self, counts):
        """
        Calculate the Total Variation Distance (TVD).

        Parameters:
        - counts: List or numpy array of observed counts.

        Returns:
        - tvd: Total Variation Distance.
        """
        counts = np.array(counts, dtype=np.float64)
        total_count = counts.sum()
        total_nodes = len(counts)
        uniform_count = total_count / total_nodes
        probability_distribution = np.array([uniform_count] * total_nodes, dtype=np.float64)
        prob = counts / total_count
        uniform_prob = probability_distribution / total_count
        tvd = 0.5 * np.sum(np.abs(prob - uniform_prob))
        return tvd

    def calculate_hellinger(self, counts):
        """
        Calculate the Hellinger Distance.

        Parameters:
        - counts: List or numpy array of observed counts.

        Returns:
        - hellinger_distance: Hellinger Distance.
        """
        counts = np.array(counts, dtype=np.float64)
        total_count = counts.sum()
        total_nodes = len(counts)
        uniform_count = total_count / total_nodes
        probability_distribution = np.array([uniform_count] * total_nodes, dtype=np.float64)
        counts += self.epsilon
        probability_distribution += self.epsilon
        prob = counts / counts.sum()
        uniform_prob = probability_distribution / probability_distribution.sum()
        hellinger_distance = np.sqrt(0.5 * np.sum((np.sqrt(prob) - np.sqrt(uniform_prob))**2))
        return hellinger_distance


class similarity_metric:

    def __init__(self):
        self.Embedding = Embedding()
        service_context = ServiceContext.from_defaults(embed_model=self.Embedding.get_model(), llm=None)
        self.similarity_evaluator = SemanticSimilarityEvaluator(
            service_context=service_context,
            similarity_mode=SimilarityMode.DEFAULT,
            similarity_threshold=0.8,
        )

    async def sem_similarity_metrics(self, y_pred, y_true):
        similarity_scores = []
        for response, reference in zip(y_pred, y_true):
            result = await self.similarity_evaluator.aevaluate(
                response=response,
                reference=reference,
            )
            similarity_scores.append(result.score)
        return similarity_scores
