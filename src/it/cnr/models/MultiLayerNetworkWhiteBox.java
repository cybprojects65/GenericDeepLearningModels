package it.cnr.models;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.api.FwdPassType;
import org.deeplearning4j.nn.api.layers.RecurrentLayer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.workspace.WorkspaceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MultiLayerNetworkWhiteBox extends MultiLayerNetwork {

	private static final Logger log = LoggerFactory.getLogger(MultiLayerNetworkWhiteBox.class);

	public MultiLayerNetworkWhiteBox(MultiLayerConfiguration conf) {
		super(conf);

	}

	public List<INDArray> layerOutputs;
	@Override
	protected INDArray outputOfLayerDetached(boolean train, FwdPassType fwdPassType, int layerIndex, INDArray input,
			INDArray featureMask, INDArray labelsMask, MemoryWorkspace outputWorkspace) {
		setInput(input);
		setLayerMaskArrays(featureMask, labelsMask);
		layerOutputs = null;
		layerOutputs = new ArrayList<>();
		
		/*
		 * Idea here: we want to minimize memory, and return only the final array
		 * Approach to do this: keep activations in memory only as long as we need them.
		 * In MultiLayerNetwork, the output activations of layer X are used as input to
		 * layer X+1 Which means: the workspace for layer X has to be open for both
		 * layers X and X+1 forward pass.
		 * 
		 * Here, we'll use two workspaces for activations: 1. For even index layers,
		 * activations WS that opens on start of even layer fwd pass, closes at end of
		 * odd layer fwd pass 2. For odd index layers, activations WS that opens on
		 * start of odd layer fwd pass, closes at end of even layer fwd pass
		 * 
		 * Additionally, we'll reconfigure the workspace manager for the *final* layer,
		 * so that we don't have to detach
		 */
		
		log.trace("White box NN started");
		if (outputWorkspace == null || outputWorkspace instanceof DummyWorkspace) {
			WorkspaceUtils.assertNoWorkspacesOpen("Expected no workspace active in outputOfLayerDetached", true);
		} else {
			Preconditions.checkState(outputWorkspace.isScopeActive(), "Workspace \"" + outputWorkspace.getId()
					+ "\" was provided for the network/layer outputs. When provided, this workspace must be opened before "
					+ "calling the output method; furthermore, closing the workspace is the responsibility of the user");
		}

		LayerWorkspaceMgr mgrEven;
		LayerWorkspaceMgr mgrOdd;

		WorkspaceMode wsm = train ? layerWiseConfigurations.getTrainingWorkspaceMode()
				: layerWiseConfigurations.getInferenceWorkspaceMode();
		if (wsm == WorkspaceMode.NONE) {
			mgrEven = LayerWorkspaceMgr.noWorkspaces();
			mgrOdd = mgrEven;

//Check for external workspace - doesn't make sense to have one with workspace mode NONE
			if (outputWorkspace != null && !(outputWorkspace instanceof DummyWorkspace)) {
				throw new IllegalStateException("Workspace \"" + outputWorkspace.getId()
						+ "\" was provided for the network/layer outputs, however " + (train ? "training" : "inference")
						+ " workspace mode is set to NONE. Cannot put output activations into the specified workspace if"
						+ "workspaces are disabled for the network. use getConfiguration().setTraining/InferenceWorkspaceMode(WorkspaceMode.ENABLED)");
			}
		} else {
			mgrEven = LayerWorkspaceMgr.builder()
					.with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
					.with(ArrayType.ACTIVATIONS, WS_LAYER_ACT_1, WS_LAYER_ACT_X_CONFIG)
					.with(ArrayType.INPUT, WS_LAYER_ACT_2, WS_LAYER_ACT_X_CONFIG) // Inputs should always be in the
																					// previous WS
					.with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
					.build();

			mgrOdd = LayerWorkspaceMgr.builder()
					.with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
					.with(ArrayType.ACTIVATIONS, WS_LAYER_ACT_2, WS_LAYER_ACT_X_CONFIG)
					.with(ArrayType.INPUT, WS_LAYER_ACT_1, WS_LAYER_ACT_X_CONFIG) // Inputs should always be in the
																					// previous WS
					.with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
					.build();
		}
		mgrEven.setHelperWorkspacePointers(helperWorkspaces);
		mgrOdd.setHelperWorkspacePointers(helperWorkspaces);

		MemoryWorkspace wsActCloseNext = null;
		MemoryWorkspace temp = null;
		MemoryWorkspace initialWorkspace = Nd4j.getMemoryManager().getCurrentWorkspace();

		boolean traceLog = true; //log.isTraceEnabled();

		Throwable t = null;
		try {
			for (int i = 0; i <= layerIndex; i++) {
				LayerWorkspaceMgr mgr = (i % 2 == 0 ? mgrEven : mgrOdd);

				if (traceLog) {
					log.trace("About to forward pass: {} - {}", i, layers[i].getClass().getSimpleName());
				}

//Edge case: for first layer with dropout, inputs can't be in previous workspace (as it hasn't been opened yet)
//Hence: put inputs in working memory
				if (i == 0 && wsm != WorkspaceMode.NONE) {
					mgr.setWorkspace(ArrayType.INPUT, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG);
				}

				try (MemoryWorkspace wsFFWorking = mgr.notifyScopeEntered(ArrayType.FF_WORKING_MEM)) { // Working
																										// memory:
																										// opened/closed
																										// once per
																										// layer
//Activations workspaces: opened/closed every second layer.
//So mgrEven (WS_LAYER_ACT_1) open at start of 0, 2, 4, 8; closed at end of 1, 3, 5, 7 etc
//and mgrOdd (WS_LAYER_ACT_2) opened at start of 1, 3, 5, 7; closed at end of 2, 4, 6, 8 etc
					temp = mgr.notifyScopeEntered(ArrayType.ACTIVATIONS);

//Note that because we're opening activation workspaces not in a simple nested order, we'll manually
// override the previous workspace setting. Otherwise, when we close these workspaces, the "current"
// workspace may be set to the incorrect one
					temp.setPreviousWorkspace(initialWorkspace);

					if (i == 0 && input.isAttached()) {
//Don't leverage out of async DataSetIterator workspaces
						mgr.setNoLeverageOverride(input.data().getParentWorkspace().getId());
					}

					if (getLayerWiseConfigurations().getInputPreProcess(i) != null) {
						input = getLayerWiseConfigurations().getInputPreProcess(i).preProcess(input,
								getInputMiniBatchSize(), mgr);
//Validation: Exception if invalid (bad preprocessor implementation)
						validateArrayWorkspaces(mgr, input, ArrayType.ACTIVATIONS, i, true,
								"Output of layer (inference)");
					}

					if (i == layerIndex) {
						if (outputWorkspace != null && !(outputWorkspace instanceof DummyWorkspace)) {
//Place activations in user-specified workspace
							mgr.setWorkspace(ArrayType.ACTIVATIONS, outputWorkspace.getId(),
									outputWorkspace.getWorkspaceConfiguration());
						} else {
//Final activations: should be detached
							mgr.setScopedOutFor(ArrayType.ACTIVATIONS);
						}
					}

					if (fwdPassType == FwdPassType.STANDARD) {
//Standard feed-forward case
						input = layers[i].activate(input, train, mgr);
					} else if (fwdPassType == FwdPassType.RNN_TIMESTEP) {
//rnnTimeStep case
						if (layers[i] instanceof RecurrentLayer) {
							input = ((RecurrentLayer) layers[i]).rnnTimeStep(reshapeTimeStepInput(input), mgr);
						} else if (layers[i] instanceof BaseWrapperLayer
								&& ((BaseWrapperLayer) layers[i]).getUnderlying() instanceof RecurrentLayer) {
							RecurrentLayer rl = ((RecurrentLayer) ((BaseWrapperLayer) layers[i]).getUnderlying());
							input = rl.rnnTimeStep(reshapeTimeStepInput(input), mgr);
						} else if (layers[i] instanceof MultiLayerNetwork) {
							input = ((MultiLayerNetwork) layers[i]).rnnTimeStep(reshapeTimeStepInput(input));
						} else {
							input = layers[i].activate(input, false, mgr);
						}
					} else {
						throw new IllegalArgumentException(
								"Unsupported forward pass type for this method: " + fwdPassType);
					}
					layers[i].clear();
//Validation: Exception if invalid (bad layer implementation)
					validateArrayWorkspaces(mgr, input, ArrayType.ACTIVATIONS, i, false, "Output of layer (inference)");

					if (wsActCloseNext != null) {
						wsActCloseNext.close();
					}
					wsActCloseNext = temp;
					temp = null;
				}

				if (traceLog) {
					log.trace("Completed forward pass: {} - {}", i, layers[i].getClass().getSimpleName());
				}

//Edge case: for first layer with dropout, inputs can't be in previous workspace (as it hasn't been opened yet)
//Hence: put inputs in working memory -> set back to default for next use of workspace mgr
				if (i == 0 && wsm != WorkspaceMode.NONE) {
					mgr.setWorkspace(ArrayType.INPUT, WS_LAYER_ACT_2, WS_LAYER_ACT_X_CONFIG); // Inputs should always be
																								// in the previous WS
				}
				
				layerOutputs.add(input);
			}//end forward to all layers
		} catch (Throwable t2) {
			t = t2;
		} finally {
			if (wsActCloseNext != null) {
				try {
					wsActCloseNext.close();
				} catch (Throwable t2) {
					if (t != null) {
						log.error(
								"Encountered second exception while trying to close workspace after initial exception");
						log.error("Original exception:", t);
						throw t2;
					}
				}
			}
			if (temp != null) {
//Should only be non-null on exception
				while (temp.isScopeActive()) {
//For safety, should never occur in theory: a single close() call may not be sufficient, if
// workspace scope was borrowed and not properly closed when exception occurred
					try {
						temp.close();
					} catch (Throwable t2) {
						if (t != null) {
							log.error(
									"Encountered second exception while trying to close workspace after initial exception");
							log.error("Original exception:", t);
							throw t2;
						}
					}
				}
			}

			Nd4j.getMemoryManager().setCurrentWorkspace(initialWorkspace);

			if (t != null) {
				if (t instanceof RuntimeException) {
					throw ((RuntimeException) t);
				}
				throw new RuntimeException("Error during neural network forward pass", t);
			}

			if (outputWorkspace == null || outputWorkspace instanceof DummyWorkspace) {
				WorkspaceUtils.assertNoWorkspacesOpen(
						"Expected no workspace active at the end of outputOfLayerDetached", true);
			} else {
				Preconditions.checkState(outputWorkspace.isScopeActive(), "Expected output workspace to still be open"
						+ "at end of outputOfLayerDetached, but it is closed. This suggests an implementation or layer workspace problem");
			}
		}

		return input;
	}

	private INDArray reshapeTimeStepInput(INDArray input) {
		if (input.rank() == 2) { // dynamically reshape to 3D input with one time-step.
			long[] inShape = input.shape();
			input = input.reshape(inShape[0], inShape[1], 1);
		}
		return input;
	}

}
