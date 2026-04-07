import numpy as np


class Microstructure:
    """TESA Microstructure data class. Holds all properties for AEH-FE analysis."""

    def __init__(self):
        """
        Initialize a Microstructure object with all fields set to defaults.

        All attributes are initialized to None, zero, or empty lists. They
        are populated progressively as the pipeline stages execute: Stage 1
        sets EBSD/grain data, Stage 2 sets mesh data, Stage 3 sets analysis
        results, and Stage 4 reads results for post-processing.
        """
        # AEH-FE Analysis Variables
        self.nodeCoordinates = None
        self.elementIndices = None
        self.boundaryNodePairs = None
        self.dataPointCoordinates = None
        self.dataPointEulerAngles = None
        self.dataPointPhase = None
        self.dataCoordinateSystemCorrectionMatrix = None
        self.dataCoordinateSystemCorrectionAngle = None
        self.elementHomogenizationMethod = None
        self.phaseElasticStiffnessMatrix = None
        self.phaseThermalExpansionCoefficientMatrix = None
        self.chiCharacteristicFunctions = None
        self.psiCharacteristicFunctions = None
        self.DEffectiveAEH = None
        self.betaEffectiveAEH = None
        self.alphaEffectiveAEH = None
        self.DEffectiveVoigt = None
        self.betaEffectiveVoigt = None
        self.alphaEffectiveVoigt = None
        self.DEffectiveReuss = None
        self.betaEffectiveReuss = None
        self.alphaEffectiveReuss = None
        self.DEffectiveGeoMean = None
        self.betaEffectiveGeoMean = None
        self.alphaEffectiveGeoMean = None
        self.DEffectiveSCS = None
        self.betaEffectiveSCS = None
        self.alphaEffectiveSCS = None
        self.DHat = None
        self.betaHat = None
        self.macroscaleLoadInfo = None
        self.quadraturePointCoordinates = None
        self.quadraturePointElasticStiffnesses = None
        self.quadraturePointThermalExpansionCoefficients = None
        self.quadraturePointStressTemperatureModuli = None
        self.stressStrainAnalysisType = None
        self.PhaseThermalConductivityMatrix = None
        self.thermalConductivityCharacteristicFunctions = None
        self.kappaEffectiveAEH = None
        self.kappaEffectiveVoigt = None
        self.kappaEffectiveReuss = None
        self.kappaEffectiveHill = None
        self.kappaEffectiveGeoMean = None
        self.phaseThermalConductivityMatrix = None
        self.kappaHat = None
        self.quadraturePointThermalConductivity = None
        self.HeatConductionAnalysisInfo = None
        self.PreviousMacrofieldTypeHeatConduction = None
        self.MacrofieldTypeHeatConduction = None
        self.MacrofieldComponentHeatConduction = None
        self.PreviousMacrofieldComponentHeatConduction = None
        self.MicrofieldMenuSelectionHeatConduction = None
        self.MicrofieldRBFInfoHeatConduction = None
        self.macroscaleLoadInfoHeatConduction = None
        self.MicrofieldHeatConduction = [None] * 10

        # Additional Properties
        self.MeshParameters = np.array([0.01, 1.0, 0.3])
        self.MeshingState = 0
        self.MeshingTypeState = 0
        self.CurrentMeshType = 0
        self.MacrofieldType = 1
        self.PreviousMacrofieldType = 0
        self.MacrofieldComponent = np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)
        self.PreviousMacrofieldComponent = np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)
        self.StressStrainAnalysisInfo = None
        self.PreviousSixNodeElementIndexList = None
        self.MicrofieldMenuSelection = 1
        self.Microfield = [None] * 22
        self.MicrofieldRBFInfo = None
        self.OnlyWaveSpeedAnalysis = False
        self.LastLoadSavePath = None
        self.LastLoadSaveFileName = None

        # Working Directory
        self.HomeDirectory = None

        # Analysis Directory
        self.AnalysisDirectory = None

        # Save Directory
        self.SaveDirectory = None

        # AEH directory
        self.AEHDirectory = None

        # Analysis Tab handle arrays (GUI handles - stored but not used in Python)
        self.PhaseNumberHandle = None
        self.PhaseNameHandle = None
        self.PhaseModeHandle = None
        self.PhasePropertyFilenameHandle = None
        self.PhaseBrowseHandle = None
        self.PhaseColorHandle = None

        # Microstructure color info
        self.Colors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 165 / 255, 0],
            [190 / 255, 190 / 255, 190 / 255],
        ]
        self.red = [1, 0, 0]
        self.green = [0, 1, 0]
        self.blue = [0, 0, 1]
        self.yellow = [1, 1, 0]
        self.magenta = [1, 0, 1]
        self.cyan = [0, 1, 1]
        self.orange = [1, 165 / 255, 0]
        self.gray = [190 / 255, 190 / 255, 190 / 255]

        # Initial file info
        self.Filename = None
        self.FileType = None
        self.EBSDStepSize = None          # EBSD scan step size in physical units (e.g. µm); for reporting only

        # Image data
        self.DisplayData = None
        self.DisplayDataIndices = None
        self.NumberDataPoints = None
        self.NumberPhases = None

        # Data point properties
        self.EBSDCorrectionMatrix = None
        self.DataCoordinateList = None
        self.DataEulerAngle = None
        self.DataPhase = None
        self.OriginalDataCoordinateList = None
        self.OriginalDataEulerAngle = None
        self.OriginalDataPhase = None

        # Phase properties
        self.PhaseName = None
        self.PhaseColorValue = None
        self.PhasePropertyFilename = None
        self.PhaseStiffnessMatrix = None
        self.PhaseThermalExpansionMatrix = None
        self.PhaseDensity = None
        self.PhaseVolumeFraction = None

        # Grain properties
        self.NumberGrains = None
        self.Grains = None
        self.GrainColors = None
        self.GrainsNormalized = None
        self.GrainsSmoothed = None
        self.GrainPointSeparation = None
        self.AllGrainNodes = None
        self.GrainsReduced = None
        self.AllGrainEdges = None
        self.OriginalElementGrains = None
        self.GrainPhases = None
        self.GrainAngles = None
        self.ElementPhases = None
        self.ElementGrains = None
        self.GrainHolePoints = None
        self.GrainsSmoothedReduced = None
        self.GrainJunctionPoints = None
        self.FineGrainPolylines = None
        self.GrainPolylines = None
        self.PhasePolylines = None
        self.GrainPolylinePairs = None
        self.GrainHoles = None
        self.HoleGrains = None
        self.GrainMedialAxis = None

        # Analysis properties
        self.MeshDensity = None
        self.MeshingMethod = None
        self.HomogenizationMethod = None
        self.CompletedHomogenizationMethods = [None] * 5
        self.ElementLevelHomogenizationMethodValue = None

        # FE information
        self.ThreeNodeCoordinateList = None
        self.ThreeNodeElementIndexList = None
        self.SixNodeCoordinateList = None
        self.SixNodeElementIndexList = None
        self.BoundaryNodeRelationsList = None
        self.DataPointsInElementList = None
        self.NumberElements = None
        self.NumberNodes = None
        self.MeshSizeFunctionGrid = None
        self.MeshSizeFunctionGridLimits = None
        self.InitialMesh = None
        self.GrainsElements = None
        self.GrainsMeshed = None

        # Post-analysis options and handles storage
        self.SelectedHomogenizationPlots = None
        self.SelectedHomogenizationTypes = None
        self.SelectedHomogenizationPlotTypes = None
        self.SelectedPhasePlots = None
        self.PlotWavespeedType = ['VP', 'VS1', 'VS2', 'VSH', 'VSV', 'AVS', 'DTS', 'DTP']
        self.PlotScheme = None
        self.HomogenizationCheckboxHandle = None
        self.HomogenizationTypeHandle = None
        self.HomogenizationPlotTypeHandle = None
        self.PhaseByPhaseHomogenizationMethod = None
        self.PhaseNameCheckboxHandle = None
        self.PhaseByPhasePlotChoiceValue = None
        self.PlotType = None
        self.SubPlotHandle = None
        self.SubPlotTitleBoxHandle = None
        self.SubPlotXLabelHandle = None
        self.SubPlotYLabelHandle = None
        self.SubPlotColorbarHandle = None
        self.SubPlotContourValue = None
        self.BulkStiffnessEntryHandle = None

        # Post Analysis plot annotation storage
        self.PlotData = None

        # Post-analysis information
        self.HomogenizationMethods = ['AEH', 'Voigt', 'Reuss', 'Hill', 'Geometric Mean']
        self.HomogenizedDensity = None
        self.PhaseAnisotropyMatrix = [None] * 3
        self.HomogenizedStiffnessMatrix = [None] * 5
        self.SphericalWaveSpeeds = [[None] * 11 for _ in range(5)]
        self.EqualAreaWaveSpeeds = [None] * 5
        self.SphericalPolarizationVectors = None
        self.EqualAreaPolarizationVectors = None
        self.AVP = [None] * 5
        self.AVS1 = [None] * 5
        self.AVS2 = [None] * 5
        self.AVSH = [None] * 5
        self.AVSV = [None] * 5
        self.MaxAVS = [None] * 5
        self.PhaseAnisotropySphericalWaveSpeeds = None
        self.PhaseAnisotropyEqualAreaWaveSpeeds = None
        self.PhaseAnisotropyAVP = None
        self.PhaseAnisotropyAVS1 = None
        self.PhaseAnisotropyAVS2 = None
        self.PhaseAnisotropyAVSH = None
        self.PhaseAnisotropyAVSV = None
        self.PhaseAnisotropyMaxAVS = None
        self.PhaseWaveSpeedResults = {}

        # Plot labels
        self.DefaultTitle = [None] * 8
        self.DefaultUnits = [None] * 8

        # RBF interpolants
        self.HomogenizationRBF = [[None] * 11 for _ in range(5)]
        self.PhaseRBF = None

        # Store default colormaps for sphere and equal area plot
        self.CMAPNames = [
            'JET', 'HSV', 'HOT', 'COOL', 'SPRING', 'SUMMER', 'AUTUMN',
            'WINTER', 'GRAY', 'BONE', 'COPPER', 'PINK',
        ]
        self.CMAP = None
