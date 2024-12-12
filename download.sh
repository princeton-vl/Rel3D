wgetgdrive(){
  # $1 = file ID
  # $2 = file name

  URL="https://docs.google.com/uc?export=download&id=$1"

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

mkdir tmp
key="$1"
case $key in
	data_min)
		wgetgdrive 1qebCo0vzO6zlgEzSyd85_o4RpnyzmB0P tmp/data_min.zip
		unzip -o tmp/data_min.zip
		mkdir -p data
		mv data_min/* data
		rm -rf data_min
    		;;
	data)
		wgetgdrive 1g3CXOLaouk7n1vtQXnf11B8aGmF7qPKT tmp/data.zip
		unzip -o tmp/data.zip
    		;;
	pretrained_models)
		wgetgdrive 1ICpcpyk-hfJ0gXaxbSSh1DWhOLs5DiwZ tmp/pretrained_models.zip
		unzip -o tmp/pretrained_models.zip
    		;;
	data_raw)
		wgetgdrive 1ArpgHg8uj90zfpf0ytYDW4S-K9DPLkuI tmp/20200223_selected.zip
		unzip -o tmp/20200223_selected.zip
		mkdir -p data
		mv 20200223 data
    		;;
    	*)
    		echo "unknow argument $1" # unknown argument
    		;;
esac
rm -r tmp
