#if !defined(MAIN_RETURN_TYPE)
    #define MAIN_RETURN_TYPE int
#endif

#if !defined(MAIN_RETURN_STMT)
    #define MAIN_RETURN_STMT return 0
#endif

#if !defined(SHADER_INDEX)
    #define SHADER_INDEX 0
#endif

#if !defined(SHADER_COUNT)
    #define SHADER_COUNT 1
#endif

#if !defined(FLOAT_TYPE)
    #define FLOAT_TYPE float
#endif

#if !defined(INT_TYPE)
    #define INT_TYPE int
#endif

#if !defined(BUFFER_0_SIZE)
    #define BUFFER_0_SIZE 128
#endif

#if !defined(DEFINE_BUFFER_0)
typedef struct {
    INT_TYPE   first_level;
    INT_TYPE   last_level;
    FLOAT_TYPE mass_atom_0;
    FLOAT_TYPE mass_atom_1;
    FLOAT_TYPE distance_to_asymptote;
    FLOAT_TYPE integration_step;
} buffer_0_struct;

    #define DEFINE_BUFFER_0 static buffer_0_struct buffer_0[BUFFER_0_SIZE];
#endif

#if !defined(GET_ITEM_BUFFER_0)
    #define GET_ITEM_BUFFER_0(element_index) buffer_0[BUFFER_0_SIZE * SHADER_INDEX + element_index]
#endif

#if !defined(BUFFER_1_SIZE)
    #define BUFFER_1_SIZE 128
#endif

#if !defined(DEFINE_BUFFER_1)
    #define DEFINE_BUFFER_1 static FLOAT_TYPE buffer_1[BUFFER_1_SIZE];
#endif

#if !defined(GET_ITEM_BUFFER_1)
    #define GET_ITEM_BUFFER_1(element_index) buffer_1[BUFFER_1_SIZE * SHADER_INDEX + element_index]
#endif

#if !defined(BUFFER_2_SIZE)
    #define BUFFER_2_SIZE 128
#endif

#if !defined(DEFINE_BUFFER_2)
    #define DEFINE_BUFFER_2 static FLOAT_TYPE buffer_2[BUFFER_2_SIZE];
#endif

#if !defined(BUFFER_3_SIZE)
    #define BUFFER_3_SIZE 128
#endif

#if !defined(DEFINE_BUFFER_3)
    #define DEFINE_BUFFER_3 static FLOAT_TYPE buffer_3[BUFFER_3_SIZE];
#endif

#if !defined(BUFFER_4_SIZE)
    #define BUFFER_4_SIZE 128
#endif

#if !defined(DEFINE_BUFFER_4)
    #define DEFINE_BUFFER_4 static FLOAT_TYPE buffer_4[BUFFER_4_SIZE];
#endif

#if !defined(BUFFER_5_SIZE)
    #define BUFFER_5_SIZE 128
#endif

#if !defined(DEFINE_BUFFER_5)
    #define DEFINE_BUFFER_5 static FLOAT_TYPE buffer_5[BUFFER_5_SIZE];
#endif

#if !defined(PROGRAM_HEADER)
    #define PROGRAM_HEADER
#endif
