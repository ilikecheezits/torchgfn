import jax
import jax.numpy as jnp
from jax_rnafold.common.vienna_rna import ViennaContext
from jax_rnafold.common.utils import TURNER_2004, TURNER_1999
import os

TURNER_1999 = os.path.join(os.path.dirname(__file__), "jax-rnafold", "src", "jax_rnafold", "data", "thermo-params", "rna_turner1999.par")

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
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    matrix = matrix.at[i, j].set(1)
                    matrix = matrix.at[j, i].set(1)
        
        row_sums = jnp.sum(matrix, axis=1)
        is_unpaired = (row_sums == 0)
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

if __name__ == '__main__':
    print("🛑 Week 1 Gatekeeper Test")

    seq_gatekeeper_1 = "CCCCGGGG"
    onehot_gatekeeper_1 = RNAOracle.seq_to_onehot(seq_gatekeeper_1)
    target_struct_gatekeeper = "((....))"

    mfe_gatekeeper_1 = RNAOracle.get_mfe(onehot_gatekeeper_1)
    print(f"MFE for {seq_gatekeeper_1}: {mfe_gatekeeper_1}")
    if mfe_gatekeeper_1 < 0: print("✅ MFE is negative, as expected.")
    else: print("❌ MFE is NOT negative.")

    defect_gatekeeper_1 = RNAOracle.compute_defect(onehot_gatekeeper_1, target_struct_gatekeeper)
    print(f"Defect for {seq_gatekeeper_1} with target {target_struct_gatekeeper}: {defect_gatekeeper_1}")
    if abs(defect_gatekeeper_1 - 0.0) < 1e-1:
        print("✅ Defect is close to 0, as expected.")
    else:
        print("❌ Defect is NOT close to 0.")

    seq_gatekeeper_2 = "AAAAAAAA"
    onehot_gatekeeper_2 = RNAOracle.seq_to_onehot(seq_gatekeeper_2)
    defect_gatekeeper_2 = RNAOracle.compute_defect(onehot_gatekeeper_2, target_struct_gatekeeper)
    print(f"Defect for {seq_gatekeeper_2} with target {target_struct_gatekeeper}: {defect_gatekeeper_2}")
    if abs(defect_gatekeeper_2 - 8.0) < 0.1: print("✅ Defect for AAAAAAAA has spiked, as expected.")
    else: print("❌ Defect for AAAAAAAA has NOT spiked to ~8.0.")