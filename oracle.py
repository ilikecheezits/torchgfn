import jax
import jax.numpy as jnp
from jax_rnafold.common.utils import TURNER_1999
from jax_rnafold.common.vienna_rna import ViennaContext


class RNAOracle:
    @staticmethod
    def seq_to_onehot(sequence_string: str) -> jnp.ndarray:
        """
        Converts an RNA sequence string "ACGU" to a one-hot encoded JAX array.
        """
        char_to_int = {char: i for i, char in enumerate("ACGU")}
        integers = jnp.array([char_to_int[char] for char in sequence_string])
        return jax.nn.one_hot(integers, num_classes=4)

    @staticmethod
    def onehot_to_seq(onehot_seq: jnp.ndarray) -> str:
        """
        Converts a one-hot encoded JAX array to an RNA sequence string "ACGU".
        """
        int_to_char = {i: char for i, char in enumerate("ACGU")}
        integers = jnp.argmax(onehot_seq, axis=-1)
        return "".join([int_to_char[int_val.item()] for int_val in integers])

    @staticmethod
    def _dotbracket_to_binary_matrix(dotbracket_string: str) -> jnp.ndarray:
        """
        Converts a dot-bracket string to a binary pairing matrix.
        Includes self-pairing (diagonal=1) for unpaired bases.
        """
        n = len(dotbracket_string)
        matrix = jnp.zeros((n, n), dtype=jnp.int32)
        stack = []
        for i, char in enumerate(dotbracket_string):
            if char == "(":
                stack.append(i)
            elif char == ")":
                if stack:
                    j = stack.pop()
                    matrix = matrix.at[i, j].set(1)
                    matrix = matrix.at[j, i].set(1)

        row_sums = jnp.sum(matrix, axis=1)
        is_unpaired = row_sums == 0
        diag_indices = jnp.diag_indices(n)
        matrix = matrix.at[diag_indices].add(is_unpaired.astype(jnp.int32))

        return matrix

    @staticmethod
    def get_mfe(onehot_seq: jnp.ndarray) -> float:
        seq_string = RNAOracle.onehot_to_seq(onehot_seq)
        vc = ViennaContext(seq_string, params_path=TURNER_1999)
        return vc.mfe()

    @staticmethod
    def get_partition(onehot_seq: jnp.ndarray) -> float:
        seq_string = RNAOracle.onehot_to_seq(onehot_seq)
        vc = ViennaContext(seq_string, params_path=TURNER_1999)
        pf = vc.pf()
        return pf

    @staticmethod
    def compute_defect(onehot_seq: jnp.ndarray, target_struct: str) -> float:
        """
        Computes the ensemble defect (L2 loss) between the predicted pairing probabilities
        (including unpaired probabilities) and a target structure.
        """
        seq_string = RNAOracle.onehot_to_seq(onehot_seq)
        vc = ViennaContext(seq_string, params_path=TURNER_1999)
        P = jnp.array(vc.make_bppt())
        p_unpaired = 1.0 - jnp.sum(P, axis=-1)
        n = P.shape[0]
        diag_indices = jnp.diag_indices(n)
        P = P.at[diag_indices].set(p_unpaired)
        T = RNAOracle._dotbracket_to_binary_matrix(target_struct)
        defect = jnp.sum(jnp.square(P - T))
        return float(defect)
