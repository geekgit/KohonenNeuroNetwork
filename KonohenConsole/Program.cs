using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KonohenConsole
{
    class Program
    {
        public static double Sigmoid(double s, object param)
        {

            double a = (double)param;
            double c = Math.Exp(a * s) - Math.Exp(-1 * a * s);
            double z = Math.Exp(a * s) + Math.Exp(-1 * a * s);
            double r = c / z;
            return r;
        }
        static int RndInt(Random R, int A, int B)
        {
            int rvalue = R.Next(A, B + 1);
            return rvalue;
        }
        static float RndFloat(Random R, float A, float B)
        {
            double len = Math.Abs(B - A);
            double k = R.NextDouble();
            double rlen = k * len;
            double rvalue = A + rlen;
            return (float)rvalue;
        }
        static double RndDouble(Random R, double A, double B)
        {
            double len = Math.Abs(B - A);
            double k = R.NextDouble();
            double rlen = k * len;
            double rvalue = A + rlen;
            return rvalue;
        }

        static void Kohonen(ArrayList Data,int classes, int characteristics,double alpha,double D,double Aval=0.1,double Bval=0.2)
        {
            Random R = new Random(DateTime.Now.Millisecond);
            double eps = 0.00000001;
            double[,] W = new double[classes, characteristics];
            for (int i = 0; i < classes; ++i)
            {
                for (int j = 0; j < characteristics; ++j)
                {
                    W[i, j] = RndDouble(R, Aval, Bval);
                }
            }
            Console.WriteLine("W:");
            for (int i = 0; i < W.GetLength(0); ++i)
            {
                for (int j = 0; j < W.GetLength(1); ++j)
                {
                    double w = W[i, j];
                    Console.Write(w);
                    Console.Write(" ");
                }
                Console.WriteLine();
            }
            int datalen=Data.Count;
            double[,] HiddenW = (double[,])W.Clone();
            while (true)
            {
                Console.WriteLine("iterate");
                for (int t = 0; t < datalen; ++t)
                {
                    double[] X = (double[])Data[t];
                    double mindiff = double.PositiveInfinity;
                    int mindiff_ind = -1;
                    for (int i = 0; i < classes; ++i)
                    {
                        double diff = 0;
                        for (int j = 0; j < characteristics; ++j)
                        {
                            diff += Math.Pow(X[j] - W[i, j], 2);
                        }
                        if (diff < mindiff)
                        {
                            mindiff = diff;
                            mindiff_ind = i;
                        }
                    }
                    int k = mindiff_ind;
                    for (int j = 0; j < classes; ++j)
                    {
                        W[k, j] = W[k, j] + alpha * (X[j] - W[k, j]);
                    }
                }
                double maxdiff = double.NegativeInfinity;
                for (int i = 0; i < W.GetLength(0); ++i)
                {
                    for (int j = 0; j < W.GetLength(1); ++j)
                    {
                        
                        double wdiff = Math.Abs(W[i, j] - HiddenW[i, j]);
                        if (wdiff > maxdiff) maxdiff = wdiff;
                        Console.WriteLine("{0}->{1} diff:{2}", W[i, j], HiddenW[i, j], wdiff);
                    }
                }
                if (maxdiff < eps) break;
                HiddenW = (double[,])W.Clone();
            }
            Console.WriteLine("W:");
            for (int i = 0; i < W.GetLength(0); ++i)
            {
                for (int j = 0; j < W.GetLength(1); ++j)
                {
                    double w = W[i, j];
                    Console.Write(w);
                    Console.Write(" ");
                }
                Console.WriteLine();
            }
            double[,] nresults = new double[classes, datalen];
            for (int t = 0; t < datalen; ++t)
            {
                double[] X = (double[])Data[t];
                for (int i = 0; i < classes; ++i)
                {
                    double s = 0;
                    for (int j = 0; j < characteristics; ++j)
                    {
                        double chara = X[j];
                        double w = W[i, j];
                        double wc = w * chara;
                        s += wc;
                    }
                    double fresult = Sigmoid(s, 1.0);
                    //Console.WriteLine("Нейрон {0} Вход {1} Сумма {2} Функция {3}",i, t, s, fresult);
                    nresults[i, t] = fresult;
                }
            }
            for (int n = 0; n < classes; ++n)
            {
                for (int t = 0; t < datalen; ++t)
                {
                    Console.WriteLine("Нейрон {0} Вход {1} Функция {2}", n, t, nresults[n, t]);
                }
            }
        }
        /*
        static ArrayList Normalize(int classes, ArrayList Data)
        {
        }
         */
        static void Main(string[] args)
        {
            ArrayList Data = new ArrayList();
            Data.Add(new double[] {0,1,0});
            Data.Add(new double[] {1,0,1});
            Data.Add(new double[] {2, 3,0 });

            Data.Add(new double[] { 9, 7,8 });
            Data.Add(new double[] { 10, 15,9 });
            Kohonen(Data,2, 3,0.1,4);
        }
    }
}
