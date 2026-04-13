"""MBS Prepayment app entry point.

Mirrors ``rdagent.app.data_science.loop`` but uses
``MBSPrepaymentRDLoop`` instead of the generic ``DataScienceRDLoop``,
which wires the Priority 1-10 MBS modules (orchestration, memory,
search strategy, domain validation, phase gates, persona routing)
into the live runtime.

Usage::

    rdagent mbs_prepayment --competition mbs_prepayment
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import fire

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.mbs_prepayment.loop import MBSPrepaymentRDLoop


def main(
    path: Optional[str] = None,
    checkout: bool = True,
    checkout_path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    timeout: Optional[str] = None,
    competition: str = "mbs_prepayment",
    replace_timer: bool = True,
    exp_gen_cls: Optional[str] = None,
):
    """Launch the MBS Prepayment RD loop.

    Parameters
    ----------
    path :
        Restore from a session checkpoint, e.g.
        ``$LOG_PATH/__session__/1/0_propose``.
    checkout :
        If True, clear future session logs when restoring.
    checkout_path :
        Save the restored session to this path instead of overwriting.
    step_n :
        Max steps to run (None = unlimited).
    loop_n :
        Max loops to run (None = unlimited).
    timeout :
        Wall-clock duration limit (e.g. ``"2h"``).
    competition :
        Data folder name under ``DS_LOCAL_DATA_PATH``.
    replace_timer :
        Replace the timer with session timer when loading.
    exp_gen_cls :
        Optional override for the proposal generator class.
    """
    if checkout_path is not None:
        checkout = Path(checkout_path)

    if competition is not None:
        DS_RD_SETTING.competition = competition

    if not DS_RD_SETTING.competition:
        logger.error("Please specify competition name.")

    # Force the MBS scenario class regardless of whatever DS_SCEN may be set
    # to in the environment. The dedicated ``rdagent mbs_prepayment`` command
    # must be self-contained: MBSPrepaymentRDLoop.__init__ requires a scen
    # with ``mbs_orchestrator`` and will fail hard otherwise.
    mbs_scen_path = "rdagent.scenarios.mbs_prepayment.scenario.MBSPrepaymentScen"
    if DS_RD_SETTING.scen != mbs_scen_path:
        logger.info(
            f"mbs_prepayment: overriding DS_SCEN ({DS_RD_SETTING.scen}) "
            f"→ {mbs_scen_path}"
        )
        DS_RD_SETTING.scen = mbs_scen_path

    if path is None:
        mbs_loop = MBSPrepaymentRDLoop(DS_RD_SETTING)
    else:
        mbs_loop = MBSPrepaymentRDLoop.load(
            path, checkout=checkout, replace_timer=replace_timer
        )

    if exp_gen_cls is not None:
        mbs_loop.exp_gen = import_class(exp_gen_cls)(mbs_loop.exp_gen.scen)

    asyncio.run(mbs_loop.run(step_n=step_n, loop_n=loop_n, all_duration=timeout))


if __name__ == "__main__":
    fire.Fire(main)
