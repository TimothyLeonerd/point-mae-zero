# from .runner import run_net
from .runner import test_net
from .runner_pretrain import run_net as pretrain_run_net
from .runner_finetune import run_net as finetune_run_net
# from .runner_finetune import run_linear_net as linear_run_net
from .runner_finetune import test_net as test_run_net
from .runner_linear_probing import run_net as linear_probing_run_net