from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.utils.utils import EnvConfig
from pathlib import Path
from jsonargparse import CLI

ROOT_FOLDER = Path(__file__).parent.parent
RESULTS_FOlDER = ROOT_FOLDER / "results"

def eval(
    tasks: str,
    model_name: str,
):

    evaluation_tracker = EvaluationTracker(
        output_dir=RESULTS_FOlDER.as_posix(),
        save_details=True,
    )
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        env_config=EnvConfig(cache_dir="tmp/"),
        override_batch_size=1,
    )
    model_config = TransformersModelConfig(pretrained=model_name)

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

if __name__ == "__main__":
    CLI(eval)
