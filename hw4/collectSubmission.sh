files="VAE.ipynb"

for file in $files
do
    if [ ! -f $file ]; then
        echo "Required notebook $file not found."
        exit 0
    fi
done

rm -f hw4_results.zip
zip -r hw4_results.zip . -x "*.git*" "hw4_results*"  "*cs231n/datasets*" "*.ipynb_checkpoints*" "*collectSubmission.sh" "*requirements.txt" ".env/*" "*.pyc" "*cs231n/build/*"
