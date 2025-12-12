import jax
import jax.numpy as jnp
from jax_rnafold.common.vienna_rna import ViennaContext
from jax_rnafold.common.utils import TURNER_1999

class RNAOracle:
    """
    Oracle for RNA sequences.
    """

    @staticmethod
    def seq_to_onehot(sequence_string: str) -> jnp.ndarray:
        """
        Converts an RNA sequence string "ACGU" to a one-hot encoded JAX array.

        Args:
            sequence_string: The RNA sequence string.

        Returns:
            A JAX array of shape (L, 4) where L is the length of the sequence.
        """
        char_to_int = {char: i for i, char in enumerate("ACGU")}
        integers = jnp.array([char_to_int[char] for char in sequence_string])
        return jax.nn.one_hot(integers, num_classes=4)

    @staticmethod
    def onehot_to_seq(onehot_seq: jnp.ndarray) -> str:
        """
        Converts a one-hot encoded JAX array to an RNA sequence string "ACGU".

        Args:
            onehot_seq: A JAX array of shape (L, 4) with one-hot encoding.

        Returns:
            The RNA sequence string.
        """
        int_to_char = {i: char for i, char in enumerate("ACGU")}
        integers = jnp.argmax(onehot_seq, axis=-1)
        return "".join([int_to_char[int_val.item()] for int_val in integers])

    @staticmethod
    def get_mfe(onehot_seq: jnp.ndarray) -> float:
        """
        Calls rnafold.mfe to get the Minimum Free Energy (MFE) of an RNA sequence.

        Args:
            onehot_seq: A JAX array of shape (L, 4) with one-hot encoding.

        Returns:
            The MFE as a scalar float.
        """
        seq_string = RNAOracle.onehot_to_seq(onehot_seq)
        vc = ViennaContext(seq_string, params_path=TURNER_1999)
        return vc.mfe()

    @staticmethod
    def get_partition(onehot_seq: jnp.ndarray) -> float:
        """
        Calls rnafold.partition to get the partition function of an RNA sequence.

        Args:
            onehot_seq: A JAX array of shape (L, 4) with one-hot encoding.

        Returns:
            The partition function as a scalar float.
        """
        seq_string = RNAOracle.onehot_to_seq(onehot_seq)
        vc = ViennaContext(seq_string, params_path=TURNER_1999)
        pf = vc.pf()
        return pf

if __name__ == '__main__':
    # Test seq_to_onehot and onehot_to_seq
    sequence_simple = "ACGU"
    one_hot_simple = RNAOracle.seq_to_onehot(sequence_simple)
    seq_from_onehot = RNAOracle.onehot_to_seq(one_hot_simple)
    assert seq_from_onehot == sequence_simple
    print("seq_to_onehot and onehot_to_seq tests passed!")

    # Test MFE and Partition Function for a simple sequence
    sequence_mfe_pf_simple = "GCGC"
    onehot_mfe_pf_simple = RNAOracle.seq_to_onehot(sequence_mfe_pf_simple)
    expected_mfe_simple = 0.0
    expected_partition_simple = 1.0 

    calculated_mfe_simple = RNAOracle.get_mfe(onehot_mfe_pf_simple)
    calculated_partition_simple = RNAOracle.get_partition(onehot_mfe_pf_simple)

    print(f"Simple Calculated MFE: {calculated_mfe_simple:.15f}")
    print(f"Simple Calculated Partition Function: {calculated_partition_simple:.15f}")

    assert abs(calculated_mfe_simple - expected_mfe_simple) < 1e-6
    assert abs(calculated_partition_simple - expected_partition_simple) < 1e-6
    print("Simple MFE and Partition Function tests passed!")

    # Test MFE and Partition Function for a more complex sequence
    sequence_complex = "GGGGCCCC"
    onehot_complex = RNAOracle.seq_to_onehot(sequence_complex)
    expected_mfe_complex = -0.600000023841858
    expected_partition_complex = 3.924516522147465

    calculated_mfe_complex = RNAOracle.get_mfe(onehot_complex)
    calculated_partition_complex = RNAOracle.get_partition(onehot_complex)

    print(f"Complex Calculated MFE: {calculated_mfe_complex:.15f}")
    print(f"Complex Calculated Partition Function: {calculated_partition_complex:.15f}")

    assert abs(calculated_mfe_complex - expected_mfe_complex) < 1e-6
    assert abs(calculated_partition_complex - expected_partition_complex) < 1e-6

    print("\nComplex MFE and Partition Function tests passed!")

    # Test MFE and Partition Function for a random sequence
    sequence_random = "ACUAUAGUCC"
    onehot_random = RNAOracle.seq_to_onehot(sequence_random)
    expected_mfe_random = 0.0
    expected_partition_random = 1.0448533789006789

    calculated_mfe_random = RNAOracle.get_mfe(onehot_random)
    calculated_partition_random = RNAOracle.get_partition(onehot_random)

    print(f"\nRandom Calculated MFE: {calculated_mfe_random:.15f}")
    print(f"Random Calculated Partition Function: {calculated_partition_random:.15f}")

    assert abs(calculated_mfe_random - expected_mfe_random) < 1e-6
    assert abs(calculated_partition_random - expected_partition_random) < 1e-6
    print("Random MFE and Partition Function tests passed!")

    print("\nAll tests passed!")