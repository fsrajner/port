using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application2
{
    class Program
    {
        static void Main(string[] args)
        {
            // Ctrl + . -al elő lehet hozni az include-ot
            Stopwatch sw = Stopwatch.StartNew();
            string s = "alma";
            StringBuilder sb = new StringBuilder("alma");

            for (int i =0; i< 1000000; i++)
            {
                //s+="alma"; nagyon lassú a stringbuilder-hez képest
                sb.Append("alma");
            }
            s = sb.ToString();
            Console.WriteLine(sw.Elapsed);
        }
    }
}
