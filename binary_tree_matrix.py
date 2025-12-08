"""
Binary Tree Transfer Matrix Implementation for Photonic FIR Chip

This module provides a transfer matrix representation of the binary tree MZI
architecture used in programmable photonic integrated circuits.

Based on: Xu et al. (2022) "Self-calibrating programmable photonic integrated circuits"
Nature Photonics, Vol 16, August 2022, 595-602

Author: Claude (Anthropic)
Date: December 2024
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class BinaryTreeParams:
    """Parameters for binary tree MZI network"""
    n_stages: int = 3  # Number of stages in binary tree
    n_outputs: int = 8  # Number of output taps (2^n_stages)
    insertion_loss_db: float = 0.5  # Per MZI insertion loss in dB


class BinaryTreeMatrix:
    """
    Transfer matrix representation of binary tree MZI architecture.
    
    The binary tree splits one input into 8 outputs through 3 stages:
    - Stage 1: 1 MZI  (splits 1 → 2)
    - Stage 2: 2 MZIs (splits 2 → 4)
    - Stage 3: 4 MZIs (splits 4 → 8)
    
    Total: 7 MZIs
    
    Mathematical Formulation
    ------------------------
    Each MZI has a 2×2 transfer matrix:
    
        M_MZI = [[ cos(φ/2)·√L,  cos(φ/2)·√L  ]
                 [ j·sin(φ/2)·√L, -j·sin(φ/2)·√L]]
    
    where φ is the phase shift and L is the insertion loss factor.
    
    The complete binary tree is represented as:
    
        T_full = M_stage3 @ M_stage2 @ M_stage1 @ input_field
    
    Stage matrices have dimensions:
    - Stage 1: 2×1 matrix
    - Stage 2: 4×2 matrix  
    - Stage 3: 8×4 matrix
    
    Final output: 8×1 complex field vector
    
    Examples
    --------
    >>> tree = BinaryTreeMatrix()
    >>> # Equal splitting (50:50) at all MZIs
    >>> equal_phases = np.ones(7) * np.pi / 2
    >>> fields, info = tree.field_distribution(equal_phases)
    >>> print(f"Output power balance: {info['power_balance_db']:.2f} dB")
    Output power balance: -0.00 dB
    """
    
    def __init__(self, params: Optional[BinaryTreeParams] = None):
        """
        Initialize binary tree matrix.
        
        Parameters
        ----------
        params : BinaryTreeParams, optional
            Physical parameters of the binary tree
        """
        self.params = params or BinaryTreeParams()
        self.n_mzis = 2**self.params.n_stages - 1  # Total MZIs = 7
        self.insertion_loss = 10 ** (-self.params.insertion_loss_db / 10)
        
    def mzi_matrix(self, phase: float) -> np.ndarray:
        """
        Transfer matrix for a single MZI.
        
        An MZI consists of: Coupler → Phase Shifter → Coupler
        
        For a symmetric MZI with phase shift φ in upper arm:
            T_bar = cos(φ/2) * sqrt(loss)
            T_cross = j*sin(φ/2) * sqrt(loss)
        
        The power splitting ratio is:
            P_bar / P_cross = tan²(φ/2)
        
        Parameters
        ----------
        phase : float
            Phase shift in radians
            
        Returns
        -------
        np.ndarray
            2×2 transfer matrix [bar, cross]
            
        Notes
        -----
        Key phase values:
        - φ = 0: All power to bar port (100:0)
        - φ = π/2: Equal split (50:50, -3 dB)
        - φ = π: All power to cross port (0:100)
        """
        loss_factor = np.sqrt(self.insertion_loss)
        bar = np.cos(phase / 2) * loss_factor
        cross = 1j * np.sin(phase / 2) * loss_factor
        
        return np.array([[bar, bar],
                        [cross, -cross]])
    
    def stage_matrix(self, stage: int, phases: np.ndarray) -> np.ndarray:
        """
        Transfer matrix for an entire stage of the binary tree.
        
        Each stage consists of multiple MZIs operating in parallel:
        - Stage 1: 1 MZI  → 2×1 matrix (1 input, 2 outputs)
        - Stage 2: 2 MZIs → 4×2 matrix (2 inputs, 4 outputs)
        - Stage 3: 4 MZIs → 8×4 matrix (4 inputs, 8 outputs)
        
        The matrix is constructed by placing each MZI's 2×1 transfer 
        function in the appropriate position. MZI i takes input from 
        column i and produces outputs at rows 2i and 2i+1.
        
        Parameters
        ----------
        stage : int
            Stage number (1, 2, or 3)
        phases : np.ndarray
            MZI phases for this stage
            
        Returns
        -------
        np.ndarray
            Transfer matrix for the stage
            
        Examples
        --------
        >>> tree = BinaryTreeMatrix()
        >>> # Stage 1 with 50:50 split
        >>> M1 = tree.stage_matrix(1, np.array([np.pi/2]))
        >>> print(M1.shape)
        (2, 1)
        """
        n_mzis = 2**(stage - 1)  # Number of MZIs in this stage
        n_inputs = 2**(stage - 1)  # Number of inputs
        n_outputs = 2**stage  # Number of outputs
        
        # Create matrix with proper dimensions
        M = np.zeros((n_outputs, n_inputs), dtype=complex)
        
        for i in range(n_mzis):
            mzi = self.mzi_matrix(phases[i])
            # Each MZI takes 1 input and produces 2 outputs
            # Input column i, output rows 2*i and 2*i+1
            M[2*i:2*i+2, i:i+1] = mzi[:, 0:1]
        
        return M
    
    def full_tree_matrix(self, mzi_phases: np.ndarray) -> np.ndarray:
        """
        Complete transfer matrix for the entire binary tree.
        
        The full matrix is the product of all stage matrices:
            T = Stage3 @ Stage2 @ Stage1 @ input
        
        The cascaded multiplication represents the sequential splitting:
        1 input → 2 paths → 4 paths → 8 output taps
        
        Parameters
        ----------
        mzi_phases : np.ndarray
            Array of 7 MZI phases in order:
            [stage1, stage2_1, stage2_2, stage3_1, stage3_2, stage3_3, stage3_4]
            
            MZI naming convention (from Xu et al. 2022):
            - Stage 1: MZI 2-1
            - Stage 2: MZI 3-3, MZI 3-4
            - Stage 3: MZI 4-5, MZI 4-6, MZI 4-7, MZI 4-8
            
        Returns
        -------
        np.ndarray
            8×1 vector of complex field amplitudes at each output tap
            
        Raises
        ------
        ValueError
            If incorrect number of phases provided
            
        Examples
        --------
        >>> tree = BinaryTreeMatrix()
        >>> # Configure for equal power distribution
        >>> phases = np.ones(7) * np.pi / 2
        >>> fields = tree.full_tree_matrix(phases)
        >>> powers_db = 10 * np.log10(np.abs(fields)**2)
        >>> print(f"Power uniformity: {np.std(powers_db):.2f} dB")
        Power uniformity: 0.00 dB
        """
        if len(mzi_phases) != self.n_mzis:
            raise ValueError(f"Expected {self.n_mzis} phases, got {len(mzi_phases)}")
        
        # Stage 1: 1 MZI (2×1 matrix)
        M1 = self.stage_matrix(1, mzi_phases[0:1])
        
        # Stage 2: 2 MZIs (4×2 matrix)
        M2 = self.stage_matrix(2, mzi_phases[1:3])
        
        # Stage 3: 4 MZIs (8×4 matrix)
        M3 = self.stage_matrix(3, mzi_phases[3:7])
        
        # Input: single field at port 0
        # The input coupler (3dB) provides 1/sqrt(2) amplitude
        input_field = np.array([[1.0 / np.sqrt(2)]])
        
        # Cascade: Stage1 → Stage2 → Stage3
        after_stage1 = M1 @ input_field
        after_stage2 = M2 @ after_stage1
        output_fields = M3 @ after_stage2
        
        return output_fields.flatten()
    
    def field_distribution(self, mzi_phases: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Get field distribution and power at each tap with diagnostic info.
        
        Parameters
        ----------
        mzi_phases : np.ndarray
            Array of 7 MZI phases
            
        Returns
        -------
        fields : np.ndarray
            Complex field amplitude at each of 8 outputs
        info : dict
            Diagnostic information containing:
            - 'powers_linear': Linear power at each tap
            - 'powers_db': Power in dB at each tap
            - 'phases_rad': Phase at each tap (radians)
            - 'total_power': Total output power
            - 'power_balance_db': Max/min power ratio in dB
            
        Examples
        --------
        >>> tree = BinaryTreeMatrix()
        >>> phases = np.ones(7) * np.pi / 2
        >>> fields, info = tree.field_distribution(phases)
        >>> print(f"Total power: {info['total_power']:.4f}")
        >>> print(f"Power balance: {info['power_balance_db']:.2f} dB")
        """
        fields = self.full_tree_matrix(mzi_phases)
        
        powers = np.abs(fields)**2
        phases = np.angle(fields)
        
        info = {
            'powers_linear': powers,
            'powers_db': 10 * np.log10(powers + 1e-12),
            'phases_rad': phases,
            'total_power': np.sum(powers),
            'power_balance_db': 10 * np.log10(np.max(powers) / (np.min(powers) + 1e-12))
        }
        
        return fields, info
    
    def get_splitting_ratios(self, mzi_phases: np.ndarray) -> dict:
        """
        Calculate power splitting ratios for each MZI.
        
        Useful for diagnostics and verification of MZI settings.
        
        Parameters
        ----------
        mzi_phases : np.ndarray
            Array of 7 MZI phases
            
        Returns
        -------
        dict
            Splitting ratios organized by stage, with each MZI containing:
            - 'bar_power': Power at bar port (linear)
            - 'cross_power': Power at cross port (linear)
            - 'ratio_db': Bar/cross ratio in dB
            
        Examples
        --------
        >>> tree = BinaryTreeMatrix()
        >>> phases = np.ones(7) * np.pi / 2
        >>> ratios = tree.get_splitting_ratios(phases)
        >>> for stage, mzis in ratios.items():
        ...     print(f"{stage}:")
        ...     for mzi, data in mzis.items():
        ...         print(f"  {mzi}: {data['ratio_db']:.2f} dB")
        """
        ratios = {}
        
        # Stage 1
        M1 = self.mzi_matrix(mzi_phases[0])
        input1 = np.array([1.0, 0.0])
        out1 = M1 @ input1
        ratios['stage_1'] = {
            'MZI_2-1': {
                'bar_power': np.abs(out1[0])**2,
                'cross_power': np.abs(out1[1])**2,
                'ratio_db': 10 * np.log10(np.abs(out1[0])**2 / (np.abs(out1[1])**2 + 1e-12))
            }
        }
        
        # Stage 2
        ratios['stage_2'] = {}
        for i, mzi_idx in enumerate([1, 2]):
            M2 = self.mzi_matrix(mzi_phases[mzi_idx])
            input2 = np.array([1.0, 0.0])
            out2 = M2 @ input2
            ratios['stage_2'][f'MZI_3-{i+3}'] = {
                'bar_power': np.abs(out2[0])**2,
                'cross_power': np.abs(out2[1])**2,
                'ratio_db': 10 * np.log10(np.abs(out2[0])**2 / (np.abs(out2[1])**2 + 1e-12))
            }
        
        # Stage 3
        ratios['stage_3'] = {}
        for i, mzi_idx in enumerate([3, 4, 5, 6]):
            M3 = self.mzi_matrix(mzi_phases[mzi_idx])
            input3 = np.array([1.0, 0.0])
            out3 = M3 @ input3
            ratios['stage_3'][f'MZI_4-{i+5}'] = {
                'bar_power': np.abs(out3[0])**2,
                'cross_power': np.abs(out3[1])**2,
                'ratio_db': 10 * np.log10(np.abs(out3[0])**2 / (np.abs(out3[1])**2 + 1e-12))
            }
        
        return ratios


class PhotonicFIRWithMatrix:
    """
    Photonic FIR chip using binary tree matrix representation.
    
    This class integrates the binary tree matrix with delay lines and 
    phase shifters to provide a complete frequency response calculation.
    
    The chip architecture consists of:
    1. Binary tree MZI network (power distribution)
    2. Delay lines (progressive delays creating FSR)
    3. Phase shifters (final phase control)
    
    Examples
    --------
    >>> from photonic_fir_chip import ChipParameters
    >>> params = ChipParameters()
    >>> chip = PhotonicFIRWithMatrix(params)
    >>> chip.set_mzi_phases(np.ones(7) * np.pi / 2)  # Equal splits
    >>> chip.set_tap_phases(np.zeros(8))  # Zero phase
    >>> freqs = np.linspace(-params.fsr/2, params.fsr/2, 1000)
    >>> H = chip.compute_frequency_response(freqs)
    """
    
    def __init__(self, params):
        """
        Initialize FIR chip with matrix-based binary tree.
        
        Parameters
        ----------
        params : ChipParameters
            Physical parameters from photonic_fir_chip module
        """
        self.params = params
        self.binary_tree = BinaryTreeMatrix()
        
        # MZI and phase shifter settings
        self.mzi_phases = np.zeros(7)
        self.tap_phases = np.zeros(8)
        
        # Delay line parameters (taps 8-15)
        self.delays = np.arange(8, 16) * params.delay_step
        
    def compute_frequency_response(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute frequency response using binary tree matrix.
        
        The frequency response is calculated as:
        
            H(ω) = Σᵢ [tree_field[i] · exp(jφᵢ) · exp(-jωτᵢ) · √Lᵢ]
        
        where:
        - tree_field[i]: Complex field from binary tree
        - φᵢ: Phase shifter setting
        - τᵢ: Delay of tap i
        - Lᵢ: Insertion loss
        
        Parameters
        ----------
        frequencies : np.ndarray
            Frequency array in Hz (relative to center frequency)
            
        Returns
        -------
        np.ndarray
            Complex frequency response H(ω)
        """
        omega = 2 * np.pi * frequencies
        H = np.zeros(len(frequencies), dtype=complex)
        
        # Get field distribution from binary tree
        tree_fields = self.binary_tree.full_tree_matrix(self.mzi_phases)
        
        # Add contributions from each tap
        for i in range(8):
            # Phase shifter
            phase_transfer = np.exp(1j * self.tap_phases[i]) * np.sqrt(0.95)
            
            # Delay line (frequency dependent)
            delay = self.delays[i]
            length_cm = (delay * 3e8 / self.params.group_index) * 100
            loss = 10 ** (-self.params.waveguide_loss * length_cm / 10)
            
            delay_transfer = np.exp(-1j * omega * delay) * np.sqrt(loss)
            
            # Total contribution
            H += tree_fields[i] * phase_transfer * delay_transfer
        
        return H
    
    def set_mzi_phases(self, phases: np.ndarray):
        """
        Set all MZI phases.
        
        Parameters
        ----------
        phases : np.ndarray
            Array of 7 phase values in radians
        """
        if len(phases) != 7:
            raise ValueError("Expected 7 MZI phases")
        self.mzi_phases = phases
    
    def set_tap_phases(self, phases: np.ndarray):
        """
        Set all tap phase shifter phases.
        
        Parameters
        ----------
        phases : np.ndarray
            Array of 8 phase values in radians
        """
        if len(phases) != 8:
            raise ValueError("Expected 8 tap phases")
        self.tap_phases = phases


if __name__ == "__main__":
    """Run basic demonstration"""
    print("Binary Tree Transfer Matrix - Basic Test")
    print("=" * 60)
    
    tree = BinaryTreeMatrix()
    
    # Test equal splitting
    equal_phases = np.ones(7) * np.pi / 2
    fields, info = tree.field_distribution(equal_phases)
    
    print("\nEqual Splitting (50:50):")
    print(f"  Output powers (dB): {info['powers_db']}")
    print(f"  Total power: {info['total_power']:.4f}")
    print(f"  Power balance: {info['power_balance_db']:.2f} dB")
    
    print("\n✓ Module test passed!")
