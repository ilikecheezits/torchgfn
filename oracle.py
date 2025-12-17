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

    # Helper function for single sequence MFE calculation
    @staticmethod
    def _get_mfe_single(onehot_seq: jnp.ndarray) -> jnp.ndarray:
        seq_string = RNAOracle.onehot_to_seq(onehot_seq)
        vc = ViennaContext(seq_string, params_path=str(TURNER_1999))
        return vc.mfe()

    # Helper function for single sequence Partition Function calculation
    @staticmethod
    def _get_partition_single(onehot_seq: jnp.ndarray) -> jnp.ndarray:
        seq_string = RNAOracle.onehot_to_seq(onehot_seq)
        vc = ViennaContext(seq_string, params_path=str(TURNER_1999))
        pf = vc.pf()
        return pf

    # Helper function for single sequence defect calculation
    @staticmethod
    def _compute_defect_single(onehot_seq: jnp.ndarray, target_struct: str) -> jnp.ndarray:
        """
        Computes the ensemble defect (L2 loss) between the predicted pairing probabilities
        (including unpaired probabilities) and a target structure for a single sequence.
        """
        seq_string = RNAOracle.onehot_to_seq(onehot_seq)
        vc = ViennaContext(seq_string, params_path=str(TURNER_1999))
        P = jnp.array(vc.make_bppt())
        p_unpaired = 1.0 - jnp.sum(P, axis=-1)
        n = P.shape[0]
        diag_indices = jnp.diag_indices(n)
        P = P.at[diag_indices].set(p_unpaired)
        T = RNAOracle._dotbracket_to_binary_matrix(target_struct)
        defect = jnp.sum(jnp.square(P - T))
        return defect

    @staticmethod
    def get_mfe(batch_onehot_seq: jnp.ndarray) -> jnp.ndarray:
        """
        Computes MFE for a batch of one-hot encoded sequences.
        Expected input shape: (batch_size, sequence_length, 4)
        """
        mfes = [RNAOracle._get_mfe_single(seq) for seq in batch_onehot_seq]
        return jnp.array(mfes)

    @staticmethod
    def get_partition(batch_onehot_seq: jnp.ndarray) -> jnp.ndarray:
        """
        Computes partition function for a batch of one-hot encoded sequences.
        Expected input shape: (batch_size, sequence_length, 4)
        """
        partitions = [RNAOracle._get_partition_single(seq) for seq in batch_onehot_seq]
        return jnp.array(partitions)

    @staticmethod
    def compute_defect(batch_onehot_seq: jnp.ndarray, batch_target_struct: list[str]) -> jnp.ndarray:
        """
        Computes the ensemble defect for a batch of one-hot encoded sequences
        and a corresponding batch of target structures.
        Expected input shapes:
            batch_onehot_seq: (batch_size, sequence_length, 4)
            batch_target_struct: List of strings, length batch_size
        """
        defects = [
            RNAOracle._compute_defect_single(seq, target)
            for seq, target in zip(batch_onehot_seq, batch_target_struct)
        ]
        return jnp.array(defects)
