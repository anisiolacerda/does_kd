train_erm_teachers: train_erm_wilds_camelyon train_erm_terra_incognita train_erm_officehome train_erm_pacs train_erm_nico train_erm_celeba train_erm_colored_mnist
#train_erm_colored_mnist  train_erm_nico ## train teacher architecture BLA on all OoD data
n_hparams = 2
n_trials = 4
teacher_arch = 'resnet101'

train_erm_colored_mnist: 
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/ColoredMNIST_IRM/run.sh 'launch' 'local' '/srv/anisio/does_kd/data' 'ERM' $(n_hparams) $(n_trials) $(teacher_arch)

train_erm_officehome: 
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/OfficeHome/run.sh 'launch' 'local' '/srv/anisio/does_kd/data' 'ERM'  $(n_hparams) $(n_trials) $(teacher_arch)

train_erm_pacs: 
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/PACS/run.sh 'launch' 'local' '/srv/anisio/does_kd/data' 'ERM'  $(n_hparams) $(n_trials) $(teacher_arch)

train_erm_terra_incognita: 
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/TerraIncognita/run.sh 'launch' 'local' '/srv/anisio/does_kd/data' 'ERM' $(n_hparams) $(n_trials) $(teacher_arch)

train_erm_wilds_camelyon: 
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/WILDSCamelyon/run.sh 'launch' 'local' '/srv/anisio/does_kd/data' 'ERM' $(n_hparams) $(n_trials) $(teacher_arch)

train_erm_celeba: 
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/CelebA_Blond/run.sh 'launch' 'local' '/srv/anisio/does_kd/data' 'ERM'  $(n_hparams) $(n_trials) $(teacher_arch)

train_erm_nico:
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/NICO_Mixed/run.sh 'launch' 'local' '/srv/anisio/does_kd/data' 'ERM'  $(n_hparams) $(n_trials) $(teacher_arch)

clean_erm_teachers: clean_erm_coloredmnist_irm clean_erm_officehome\
 clean_erm_pacs clean_erm_terra_incognita clean_erm_wilds_camelyon   ## clean_teachers_erm_nico clean all incompleted sweeps
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/CelebA_Blond/run.sh 'delete_incomplete' 'local' '/srv/anisio/does_kd/data' 'ERM' $(n_hparams) $(n_trials) $(teacher_arch);\

clean_erm_coloredmnist_irm:
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/ColoredMNIST_IRM/run.sh 'delete_incomplete' 'local' '/srv/anisio/does_kd/data' 'ERM' $(n_hparams) $(n_trials) $(teacher_arch);\

clean_erm_celeba:
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/CelebA_Blond/run.sh 'delete_incomplete' 'local' '/srv/anisio/does_kd/data' 'ERM' $(n_hparams) $(n_trials) $(teacher_arch);\

clean_erm_nico:
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/NICO_Mixed/run.sh 'delete_incomplete' 'local' '/srv/anisio/does_kd/data' 'ERM' $(n_hparams) $(n_trials) $(teacher_arch);\

clean_erm_officehome:
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/OfficeHome/run.sh 'delete_incomplete' 'local' '/srv/anisio/does_kd/data' 'ERM' $(n_hparams) $(n_trials) $(teacher_arch);\

clean_erm_pacs:
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/PACS/run.sh 'delete_incomplete' 'local' '/srv/anisio/does_kd/data' 'ERM' $(n_hparams) $(n_trials) $(teacher_arch);\

clean_erm_terra_incognita:
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/TerraIncognita/run.sh 'delete_incomplete' 'local' '/srv/anisio/does_kd/data' 'ERM' $(n_hparams) $(n_trials) $(teacher_arch);\

clean_erm_wilds_camelyon:
	cd /srv/anisio/does_kd/external/OoD-Bench/external/DomainBed;\
	sh sweep/WILDSCamelyon/run.sh 'delete_incomplete' 'local' '/srv/anisio/does_kd/data' 'ERM' $(n_hparams) $(n_trials) $(teacher_arch)
