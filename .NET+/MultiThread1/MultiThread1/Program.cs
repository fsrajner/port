using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MultiThread1
{
    class Program
    {
        static void Main(string[] args)
        {
            var array = Enumerable.Range(2, 10000000).ToArray();
            Stopwatch sw = Stopwatch.StartNew();

            Console.WriteLine(array.AsParallel().Where( i=>isPrime(i)).Count()  );
            
            Console.WriteLine("runtime: "+sw.ElapsedMilliseconds);

          "HelloWorld"  .AsParallel()
                        .AsOrdered()
                        .Select(ch=>ch.ToString().ToUpper())
                        .ToList()
                        .ForEach(ch=>Console.Write(ch));

            Console.ReadKey();
        }

        static bool isPrime(int n)
        {
            int limit = (int)Math.Sqrt(n);

            for (int i = 2; i < limit; i++)
            {
                if (n % i == 0) return false;
            }
            return true;
        }

        static void Long(int i)
        {
            Console.WriteLine(">>"+i);
            Thread.Sleep(i * 100);
            Console.WriteLine("<<"+i);
        }
        static void Write(object param)
        {
     //       throw new NotImplementedException("hiba");
            for (int i = 0; i < 1000; i++)
            {
                Console.Write(param);
            }
        }
        static string Download(string url)
        {
            using (var Client = new WebClient()) return Client.DownloadString(url);
        }
        void old()
        {
            try
            {
                var taskegy = new Task<int>(() => 8);
                var taskketto = taskegy.ContinueWith(prev => { Thread.Sleep(1000); return prev.Result + 2; });
                var taskharom = taskketto.ContinueWith(prev => { Thread.Sleep(1000); return prev.Result + 5; });

                taskegy.Start();
                Console.WriteLine(taskegy.Result);
                Console.WriteLine(taskketto.Result);
                Console.WriteLine(taskharom.Result);
                var task1 = Task.Factory.StartNew<string>(() => Download("http://isitchristmas.net"));
                //task1.Start();
                //task.Wait();
                Console.WriteLine(task1.Result);
                var task2 = Task.Factory.StartNew<string>(() => Download("http://isitchristmas.net"));

                Console.WriteLine(task2.Result);
                var task3 = Task.Factory.StartNew<string>(() => Download("http://isitchristmas.net"));

                Console.WriteLine(task3.Result);
                ThreadPool.QueueUserWorkItem(Write, "1");
                ThreadPool.QueueUserWorkItem(Write, "2");
                ThreadPool.QueueUserWorkItem(Write, "3");

            }
            catch (Exception exc)
            {
                Console.WriteLine("ajjaj" + exc.ToString());   //tostring jobb mint a message
            }
            Console.WriteLine("End");
        }

        /* void Parallel()
        {
            // Parallel.For(100, 120, n => Long(n));

            foreach (var ch in "Hello WorldHello World")
            {
                Console.Write(ch);
            }

            Parallel.ForEach("Hello WorldHello World", ch =>
            {
                Console.Write(ch);
            }

                );
        }
         */
    }
}
