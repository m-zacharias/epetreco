TEST_INPUT_DIR="./test_input/"

make test_angular_index_counting.out

for INPUT_FILE in `ls $TEST_INPUT_DIR`; do
  ./test_angular_index_counting.out $TEST_INPUT_DIR$INPUT_FILE $INPUT_FILE
done
