import numpy as np
from voltage_ctrl import VoltageController


def main():

    chn_list = list(range(1, 17))
    volt_list = np.arange(1.0, 17.0)
    with VoltageController(
        com_port="COM3",
        baud_rate=9700,
        zero_on_exit=False,
    ) as vc:
        vc.set_voltages(
            channels=chn_list,
            voltages=volt_list,
            v_max=30,
        )


if __name__ == "__main__":
    main()
